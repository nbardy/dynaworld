#!/usr/bin/env bun

const fs = require("fs");
const http = require("http");
const path = require("path");
const puppeteer = require("puppeteer");

const ROOT_DIR = path.resolve(__dirname);
const MIME = Object.freeze({
	".css": "text/css; charset=utf-8",
	".html": "text/html; charset=utf-8",
	".js": "application/javascript; charset=utf-8",
	".json": "application/json; charset=utf-8",
	".png": "image/png",
});
const OPTION_SPECS = Object.freeze({
	"--experiment": ["experiment", String, "backward"],
	"--variant": ["variant", String, "both"],
	"--order": ["order", String, "control-first"],
	"--splats": ["splats", Number, 8192],
	"--capacity": ["capacity", Number, null],
	"--scale": ["scale", Number, 1],
	"--warmup": ["warmup", Number, 32],
	"--steps": ["steps", Number, 128],
	"--profiles": ["profiles", Number, 5],
	"--checkpoint": ["checkpoint", String, "packed-f16"],
	"--stride": ["stride", Number, 16],
	"--checkpoint-order": ["checkpointOrder", String, "pixel-major"],
	"--projection-layout": ["projectionLayout", String, "split-compact"],
	"--ssim-layout": ["ssimLayout", String, "separable"],
	"--pair-packet": ["pairPacket", String, "lane"],
	"--granularity": ["granularity", String, "checkpoint-block"],
	"--tile": ["tile", Number, 8],
	"--tile-capacity": ["tileCapacity", Number, 1024],
	"--out": ["out", String, null],
	"--port": ["port", Number, 0],
	"--timeout-ms": ["timeoutMs", Number, 180000],
	"--browser-executable": ["browserExecutable", String, null],
});

function usage() {
	return `Usage: bun web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js [options]

Runs the tiled WebGPU kernel benchmark in headless Chromium and emits JSON.
The browser is the WebGPU runtime; Bun owns orchestration and artifact output.

  --experiment MODE          backward, projection, or ssim
  --variant MODE             both, control, candidate, or a concrete variant id
  --order MODE               control-first or candidate-first
  --splats N                 active splats (default: 8192)
  --capacity N               model capacity (default: active splats)
  --scale N                  raster scale, 1..4 (default: 1)
  --warmup N                 warmup steps (default: 32)
  --steps N                  measured steps (default: 128)
  --profiles N               timestamped profile steps (default: 5)
  --checkpoint MODE          packed-f16 or f32
  --stride N                 checkpoint stride: 8, 16, or 32
  --checkpoint-order MODE    pixel-major or block-major
  --projection-layout MODE   split-compact or monolithic
  --ssim-layout MODE         separable or naive-2d
  --pair-packet MODE         lane or shared
  --granularity MODE         pair or checkpoint-block
  --tile N                   tile edge: 8 or 16
  --tile-capacity N          256, 512, 1024, 2048, or 4096
  --out PATH                 write JSON artifact instead of stdout
  --timeout-ms N             completion timeout (default: 180000)
  --browser-executable PATH  explicit Chromium executable
`;
}

function parseArgs(argv) {
	const args = Object.fromEntries(Object.values(OPTION_SPECS).map(
		([name, _convert, fallback]) => [name, fallback],
	));
	for (let index = 2; index < argv.length; index += 1) {
		const flag = argv[index];
		if (flag === "--help" || flag === "-h") return { help: true };
		const spec = OPTION_SPECS[flag];
		if (!spec) throw new Error(`Unknown option: ${flag}\n\n${usage()}`);
		const rawValue = argv[++index];
		if (rawValue === undefined) throw new Error(`${flag} requires a value.`);
		const [name, convert] = spec;
		args[name] = convert(rawValue);
		if (convert === Number && !Number.isFinite(args[name])) {
			throw new Error(`${flag} must be a finite number.`);
		}
	}
	args.capacity ??= args.splats;
	return args;
}

function executableFile(filePath) {
	if (!filePath) return false;
	try {
		fs.accessSync(filePath, fs.constants.X_OK);
		return true;
	} catch (_error) {
		return false;
	}
}

function playwrightChromiumCandidates() {
	const cacheRoot = path.join(
		process.env.HOME ?? "",
		"Library",
		"Caches",
		"ms-playwright",
	);
	if (!fs.existsSync(cacheRoot)) return [];
	return fs.readdirSync(cacheRoot)
		.filter((name) => name.startsWith("chromium-"))
		.sort()
		.reverse()
		.flatMap((name) => [
			path.join(cacheRoot, name, "chrome-mac", "Chromium.app", "Contents", "MacOS", "Chromium"),
			path.join(cacheRoot, name, "chrome-mac-arm64", "Chromium.app", "Contents", "MacOS", "Chromium"),
		]);
}

function resolveBrowserExecutable(explicitPath) {
	const candidates = [
		explicitPath,
		process.env.PUPPETEER_EXECUTABLE_PATH,
		"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
		"/Applications/Chromium.app/Contents/MacOS/Chromium",
		path.join(
			process.env.HOME ?? "",
			"Applications",
			"Google Chrome.app",
			"Contents",
			"MacOS",
			"Google Chrome",
		),
		...playwrightChromiumCandidates(),
	];
	const executable = candidates.find(executableFile);
	if (!executable) {
		throw new Error(
			"No local Chrome/Chromium executable found. Pass --browser-executable PATH.",
		);
	}
	return executable;
}

function resolveContainedPath(rootDir, requestPath) {
	let decoded;
	try {
		decoded = decodeURIComponent(requestPath);
	} catch (_error) {
		return null;
	}
	if (!decoded || decoded.includes("\0") || path.isAbsolute(decoded)) return null;
	const root = path.resolve(rootDir);
	const candidate = path.resolve(root, decoded);
	const relative = path.relative(root, candidate);
	return relative.startsWith("..") || path.isAbsolute(relative) ? null : candidate;
}

function sendFile(response, filePath) {
	fs.readFile(filePath, (error, data) => {
		if (error) {
			response.writeHead(error.code === "ENOENT" ? 404 : 500);
			response.end();
			return;
		}
		response.writeHead(200, {
			"Cache-Control": "no-store",
			"Content-Type": MIME[path.extname(filePath)] ?? "application/octet-stream",
		});
		response.end(data);
	});
}

async function startServer(port) {
	const server = http.createServer((request, response) => {
		const requestUrl = new URL(request.url, "http://127.0.0.1");
		if (requestUrl.pathname === "/favicon.ico") {
			response.writeHead(204);
			response.end();
			return;
		}
		const relative = requestUrl.pathname === "/"
			? "benchmarkTiledKernels.html"
			: requestUrl.pathname.replace(/^\/+/, "");
		const filePath = resolveContainedPath(ROOT_DIR, relative);
		if (!filePath) {
			response.writeHead(403);
			response.end();
			return;
		}
		sendFile(response, filePath);
	});
	await new Promise((resolve, reject) => {
		server.once("error", reject);
		server.listen(port, "127.0.0.1", () => {
			server.off("error", reject);
			resolve();
		});
	});
	const address = server.address();
	if (!address || typeof address === "string") {
		throw new Error("Could not determine the local benchmark server port.");
	}
	return { server, port: address.port };
}

function benchmarkUrl(port, args) {
	const query = new URLSearchParams({
		autorun: "1",
		experiment: args.experiment,
		variant: args.variant,
		order: args.order,
		splats: String(args.splats),
		capacity: String(args.capacity),
		scale: String(args.scale),
		warmup: String(args.warmup),
		steps: String(args.steps),
		profiles: String(args.profiles),
		checkpoint: args.checkpoint,
		stride: String(args.stride),
		checkpointOrder: args.checkpointOrder,
		projectionLayout: args.projectionLayout,
		ssimLayout: args.ssimLayout,
		pairPacket: args.pairPacket,
		granularity: args.granularity,
		tile: String(args.tile),
		tileCapacity: String(args.tileCapacity),
	});
	return `http://127.0.0.1:${port}/benchmarkTiledKernels.html?${query}`;
}

function delay(milliseconds) {
	return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function closeBrowser(browser) {
	if (!browser) return;
	const child = browser.process();
	await Promise.race([
		browser.close().catch(() => {}),
		delay(5000),
	]);
	if (child && child.exitCode == null) {
		child.kill("SIGTERM");
		await Promise.race([
			new Promise((resolve) => child.once("exit", resolve)),
			delay(1000),
		]);
	}
	if (child && child.exitCode == null) child.kill("SIGKILL");
}

async function runBenchmark(args) {
	const { server, port } = await startServer(args.port);
	const browserExecutable = resolveBrowserExecutable(args.browserExecutable);
	const launchOptions = {
		headless: "new",
		executablePath: browserExecutable,
		args: [
			"--enable-unsafe-webgpu",
			"--disable-dawn-features=disallow_unsafe_apis",
			"--no-sandbox",
		],
	};
	let browser;
	try {
		browser = await puppeteer.launch(launchOptions);
		const page = await browser.newPage();
		page.setDefaultNavigationTimeout(args.timeoutMs);
		page.on("console", (message) => process.stderr.write(`[page] ${message.text()}\n`));
		page.on("pageerror", (error) => process.stderr.write(`[page error] ${error.message}\n`));
		await page.setViewport({ width: 1280, height: 960 });
		await page.goto(benchmarkUrl(port, args), {
			waitUntil: "domcontentloaded",
			timeout: args.timeoutMs,
		});
		await page.waitForFunction(
			() => ["complete", "failed"].includes(
				document.documentElement.dataset.kernelBenchmarkState,
			),
			{ timeout: args.timeoutMs, polling: 100 },
		);
		const state = await page.evaluate(
			() => document.documentElement.dataset.kernelBenchmarkState,
		);
		if (state === "failed") {
			const message = await page.$eval("#kernelBenchmarkStatus", (node) => node.textContent);
			throw new Error(`Kernel benchmark failed: ${message}`);
		}
		const report = await page.evaluate(() => globalThis.__tiledKernelBenchmarkResults);
		if (!report) throw new Error("Benchmark completed without a result artifact.");
		report.execution = {
			orchestrator: "Bun",
			webGpuRuntime: "headless Chromium/Dawn",
			headless: true,
			browserExecutable,
		};
		return report;
	} finally {
		await closeBrowser(browser);
		server.closeAllConnections?.();
		await new Promise((resolve) => server.close(resolve));
	}
}

async function main() {
	const args = parseArgs(process.argv);
	if (args.help) {
		process.stdout.write(usage());
		return;
	}
	const report = await runBenchmark(args);
	const json = `${JSON.stringify(report, null, 2)}\n`;
	if (args.out) {
		const outputPath = path.resolve(args.out);
		fs.mkdirSync(path.dirname(outputPath), { recursive: true });
		fs.writeFileSync(outputPath, json);
		process.stderr.write(
			`Wrote ${report.results.length} kernel results (${report.results[0].adapter}) to ${outputPath}\n`,
		);
		return;
	}
	process.stdout.write(json);
}

main().then(
	() => process.exit(0),
	(error) => {
		process.stderr.write(`${error.stack ?? error}\n`);
		process.exit(1);
	},
);
