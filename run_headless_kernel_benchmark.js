#!/usr/bin/env bun

const fs = require("fs");
const http = require("http");
const path = require("path");
const puppeteer = require("puppeteer");

const ROOT_DIR = path.resolve(__dirname);
const PRESET_BUNDLE_PATH = path.join(ROOT_DIR, "coffee_martini_train17_holdout1.json");
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
	"--max-round-cv": ["maxRoundCv", Number, 0.10],
	"--contention-policy": ["contentionPolicy", String, "warn"],
	"--contention-sample-ms": ["contentionSampleMs", Number, 1000],
	"--postflight-cooldown-ms": ["postflightCooldownMs", Number, 10000],
	"--max-cpu-busy-fraction": ["maxCpuBusyFraction", Number, 0.85],
	"--max-load-per-cpu": ["maxLoadPerLogicalCpu", Number, 0.75],
	"--max-competing-cpu-fraction": ["maxCompetingCpuFraction", Number, 0.35],
	"--max-gpu-utilization-percent": ["maxPreflightGpuUtilizationPercent", Number, 35],
	"--min-available-memory-fraction": ["minAvailableMemoryFraction", Number, 0.10],
	"--max-swap-used-fraction": ["maxSwapUsedFraction", Number, 0.90],
	"--max-swap-to-memory-fraction": ["maxSwapUsedToMemoryFraction", Number, 0.25],
	"--out": ["out", String, null],
	"--out-dir": ["outDir", String, null],
	"--run-id": ["runId", String, null],
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
  --max-round-cv N           maximum per-variant throughput CV (default: 0.10)
  --contention-policy MODE   record, warn, or fail (default: warn)
  --contention-sample-ms N   host sample duration (default: 1000)
  --postflight-cooldown-ms N wait after owned Chrome closes (default: 10000)
  --max-cpu-busy-fraction N  pre/post CPU busy threshold (default: 0.85)
  --max-load-per-cpu N       1-minute load / logical CPU threshold (default: 0.75)
  --max-competing-cpu-fraction N
                              process CPU / host capacity threshold (default: 0.35)
  --max-gpu-utilization-percent N
                              preflight Apple GPU threshold (default: 35)
  --min-available-memory-fraction N
                              memory-pressure availability floor (default: 0.10)
  --max-swap-used-fraction N  swap occupancy ceiling (default: 0.90)
  --max-swap-to-memory-fraction N
                              used swap / physical memory ceiling (default: 0.25)
  --preflight-only            print diagnostics without launching Chrome
  --out PATH                 write JSON artifact instead of stdout
  --out-dir PATH             auto-name artifact under PATH/YYYY-MM-DD/
  --run-id ID                sanitized artifact identity for --out-dir
  --timeout-ms N             completion timeout (default: 180000)
  --browser-executable PATH  explicit Chromium executable
`;
}

function parseArgs(argv) {
	const args = Object.fromEntries(Object.values(OPTION_SPECS).map(
		([name, _convert, fallback]) => [name, fallback],
	));
	args.preflightOnly = false;
	for (let index = 2; index < argv.length; index += 1) {
		const flag = argv[index];
		if (flag === "--help" || flag === "-h") return { help: true };
		if (flag === "--preflight-only") {
			args.preflightOnly = true;
			continue;
		}
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
	if (!["record", "warn", "fail"].includes(args.contentionPolicy)) {
		throw new Error("--contention-policy must be record, warn, or fail.");
	}
	if (args.out && args.outDir) throw new Error("--out and --out-dir are mutually exclusive.");
	for (const [name, value, minimum, maximum] of [
		["--max-round-cv", args.maxRoundCv, 0.001, 1],
		["--contention-sample-ms", args.contentionSampleMs, 100, 10000],
		["--postflight-cooldown-ms", args.postflightCooldownMs, 0, 10000],
		["--max-cpu-busy-fraction", args.maxCpuBusyFraction, 0, 1],
		["--max-load-per-cpu", args.maxLoadPerLogicalCpu, 0, 10],
		["--max-competing-cpu-fraction", args.maxCompetingCpuFraction, 0, 1],
		["--max-gpu-utilization-percent", args.maxPreflightGpuUtilizationPercent, 0, 100],
		["--min-available-memory-fraction", args.minAvailableMemoryFraction, 0, 1],
		["--max-swap-used-fraction", args.maxSwapUsedFraction, 0, 1],
		["--max-swap-to-memory-fraction", args.maxSwapUsedToMemoryFraction, 0, 1],
	]) {
		if (value < minimum || value > maximum) {
			throw new Error(`${name} must be from ${minimum} through ${maximum}.`);
		}
	}
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
		maxRoundCv: String(args.maxRoundCv),
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
		page.on("requestfailed", (request) => process.stderr.write(
			`[request failed] ${request.url()} (${request.failure()?.errorText ?? "unknown"})\n`,
		));
		page.on("response", (response) => {
			if (response.status() >= 400) {
				process.stderr.write(`[response ${response.status()}] ${response.url()}\n`);
			}
		});
		await page.setViewport({ width: 1280, height: 960 });
		await page.goto(benchmarkUrl(port, args), {
			waitUntil: "domcontentloaded",
			timeout: args.timeoutMs,
		});
		try {
			await page.waitForFunction(
				() => ["complete", "failed"].includes(
					document.documentElement.dataset.kernelBenchmarkState,
				),
				{ timeout: args.timeoutMs, polling: 100 },
			);
		} catch (error) {
			const progress = await page.evaluate(() => ({
				state: document.documentElement.dataset.kernelBenchmarkState ?? "unset",
				status: document.querySelector("#kernelBenchmarkStatus")?.textContent ?? "unavailable",
			})).catch(() => ({ state: "unavailable", status: "page evaluation failed" }));
			throw new Error(
				`Kernel benchmark timed out after ${args.timeoutMs} ms `
				+ `(state=${progress.state}, status=${progress.status}).`,
				{ cause: error },
			);
		}
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

function contentionThresholds(args, resourcePlan) {
	return {
		maxCpuBusyFraction: args.maxCpuBusyFraction,
		maxLoadPerLogicalCpu: args.maxLoadPerLogicalCpu,
		maxCompetingCpuFraction: args.maxCompetingCpuFraction,
		maxPreflightGpuUtilizationPercent: args.maxPreflightGpuUtilizationPercent,
		minAvailableMemoryFraction: args.minAvailableMemoryFraction,
		minAvailableMemoryBytes: resourcePlan.minimumAvailableMemoryBytes,
		maxSwapUsedFraction: args.maxSwapUsedFraction,
		maxSwapUsedToMemoryFraction: args.maxSwapUsedToMemoryFraction,
	};
}

function readPresetMetadata() {
	const payload = JSON.parse(fs.readFileSync(PRESET_BUNDLE_PATH, "utf8"));
	const [width, height] = payload.decode_size ?? [];
	const cameras = Array.isArray(payload.cameras) ? payload.cameras : [];
	const trainViewCount = cameras.filter((camera) => camera.role === "train").length;
	for (const [label, value] of Object.entries({
		width,
		height,
		viewCount: cameras.length,
		trainViewCount,
		frameCount: payload.frame_count,
	})) {
		if (!Number.isSafeInteger(value) || value < 1) {
			throw new Error(`Preset bundle has invalid ${label}.`);
		}
	}
	return {
		width,
		height,
		viewCount: cameras.length,
		trainViewCount,
		frameCount: payload.frame_count,
	};
}

function sanitizeRunId(runId) {
	if (!runId) return null;
	const sanitized = runId
		.trim()
		.toLowerCase()
		.replace(/[^a-z0-9._-]+/g, "-")
		.replace(/^-+|-+$/g, "")
		.slice(0, 100);
	if (!sanitized) throw new Error("--run-id must contain an ASCII letter or number.");
	return sanitized;
}

function fallbackRunId(args, report) {
	return [
		report.experiment.id,
		`${args.splats}s`,
		`${report.dataset.width}x${report.dataset.height}`,
		args.order,
	].join("_");
}

function autoOutputPath(args, report, runId) {
	if (!args.outDir) return args.out ? path.resolve(args.out) : null;
	const timestamp = new Date(report.recordedAt);
	const pad = (value) => String(value).padStart(2, "0");
	const day = [
		timestamp.getFullYear(),
		pad(timestamp.getMonth() + 1),
		pad(timestamp.getDate()),
	].join("-");
	const time = [
		pad(timestamp.getHours()),
		pad(timestamp.getMinutes()),
		pad(timestamp.getSeconds()),
	].join("-");
	return path.resolve(args.outDir, day, `${time}_${runId}.json`);
}

function attachBenchmarkValidity(
	report,
	preflight,
	postflight,
	policy,
	thresholds,
	postflightCooldownMs,
) {
	const hostWarnings = [
		...preflight.assessment.warnings.map((warning) => `preflight: ${warning}`),
		...postflight.assessment.warnings.map((warning) => `postflight: ${warning}`),
	];
	const pageReasons = report.validity?.reasons ?? [];
	const correctnessAndStabilityPassed =
		report.validity?.correctnessAndStabilityPassed === true;
	report.hostDiagnostics = {
		schema: "dynaworld-benchmark-host-diagnostics/v1",
		policy,
		thresholds,
		postflightCooldownMs,
		preflight,
		postflight,
		assessment: {
			quiet: hostWarnings.length === 0,
			warnings: hostWarnings,
		},
		limitations: [
			"Quiet pre/post snapshots cannot prove the GPU was uncontended for every instant.",
			"Per-round throughput variance detects intermittent stalls but not constant-rate contention.",
			"The cooldown reduces, but cannot mathematically identify, residual driver utilization.",
			"Use matched alternating variants and repeat reversed-start runs for promotion evidence.",
		],
	};
	report.validity = {
		correctnessAndStabilityPassed,
		hostEnvironmentPassed: hostWarnings.length === 0,
		promotable: correctnessAndStabilityPassed && hostWarnings.length === 0,
		reasons: [...pageReasons, ...hostWarnings],
	};
	if (report.comparison) {
		report.comparison.validForPromotion = report.validity.promotable;
	}
}

async function main() {
	const args = parseArgs(process.argv);
	if (args.help) {
		process.stdout.write(usage());
		return;
	}
	const [{ captureHostSnapshot }, { estimateTiledBenchmarkResources }] = await Promise.all([
		import("./benchmarkHostDiagnostics.js"),
		import("./benchmarkResourcePlan.js"),
	]);
	const resourcePlan = estimateTiledBenchmarkResources(readPresetMetadata(), {
		...args,
		tileSize: args.tile,
		tileCapacity: args.tileCapacity,
		checkpointPrecision: args.checkpoint,
		checkpointStride: args.stride,
	});
	if (!resourcePlan.valid) {
		throw new Error(`Requested benchmark resource plan is invalid: ${resourcePlan.reasons.join(" ")}`);
	}
	const thresholds = contentionThresholds(args, resourcePlan);
	const preflight = await captureHostSnapshot({
		sampleMs: args.contentionSampleMs,
		thresholds,
	});
	if (!preflight.assessment.quiet && args.contentionPolicy !== "record") {
		process.stderr.write(
			`Benchmark host contention: ${preflight.assessment.warnings.join(" ")}\n`,
		);
	}
	if (args.preflightOnly) {
		process.stdout.write(`${JSON.stringify({
			schema: "dynaworld-browser-tiled-preflight/v1",
			recordedAt: new Date().toISOString(),
			options: args,
			requestedResourcePlan: resourcePlan,
			host: preflight,
			valid: preflight.assessment.quiet && resourcePlan.valid,
		}, null, 2)}\n`);
		return;
	}
	if (!preflight.assessment.quiet && args.contentionPolicy === "fail") {
		throw new Error(
			"Strict contention preflight failed. Close competing work or use "
			+ "--contention-policy warn to save a diagnostic-only run.",
		);
	}
	const report = await runBenchmark(args);
	report.requestedResourcePlan = resourcePlan;
	await delay(args.postflightCooldownMs);
	const postflight = await captureHostSnapshot({
		sampleMs: args.contentionSampleMs,
		thresholds,
	});
	attachBenchmarkValidity(
		report,
		preflight,
		postflight,
		args.contentionPolicy,
		thresholds,
		args.postflightCooldownMs,
	);
	const runId = sanitizeRunId(args.runId)
		?? (args.outDir ? fallbackRunId(args, report) : null);
	report.artifact = {
		runId,
		outputMode: args.outDir ? "auto-named" : args.out ? "explicit-path" : "stdout",
		filenameTimeZone: Intl.DateTimeFormat().resolvedOptions().timeZone,
	};
	const json = `${JSON.stringify(report, null, 2)}\n`;
	const outputPath = autoOutputPath(args, report, runId);
	if (outputPath) {
		fs.mkdirSync(path.dirname(outputPath), { recursive: true });
		fs.writeFileSync(outputPath, json);
		process.stderr.write(
			`Wrote ${report.results.length} kernel results (${report.results[0].adapter}, `
			+ `${report.validity.promotable ? "promotable" : "diagnostic only"}) `
			+ `to ${outputPath}\n`,
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
