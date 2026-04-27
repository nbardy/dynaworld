#!/usr/bin/env bun
const fs = require("fs");
const http = require("http");
const path = require("path");
const puppeteer = require("puppeteer");

function parseArgs(argv) {
	const args = {
		bundleDir: null,
		out: null,
		camera: null,
		time: 0.0,
		width: 1280,
		height: 720,
		port: 0,
	};
	for (let index = 2; index < argv.length; index += 1) {
		const arg = argv[index];
		if (arg === "--bundle-dir") {
			args.bundleDir = argv[++index];
		} else if (arg === "--out") {
			args.out = argv[++index];
		} else if (arg === "--camera") {
			args.camera = argv[++index];
		} else if (arg === "--time") {
			args.time = Number(argv[++index]);
		} else if (arg === "--width") {
			args.width = Number(argv[++index]);
		} else if (arg === "--height") {
			args.height = Number(argv[++index]);
		} else if (arg === "--port") {
			args.port = Number(argv[++index]);
		}
	}
	if (!args.bundleDir || !args.out) {
		throw new Error("--bundle-dir and --out are required.");
	}
	return args;
}

const MIME = {
	".html": "text/html",
	".js": "application/javascript",
	".css": "text/css",
	".json": "application/json",
};

function sendFile(res, filePath) {
	fs.readFile(filePath, (error, data) => {
		if (error) {
			res.writeHead(404);
			res.end();
			return;
		}
		res.writeHead(200, {
			"Content-Type":
				MIME[path.extname(filePath)] || "application/octet-stream",
		});
		res.end(data);
	});
}

async function startServer(rootDir, bundleDir, cameraPath, port) {
	const bundleRoot = path.resolve(bundleDir);
	const resolvedCameraPath = cameraPath ? path.resolve(cameraPath) : null;
	const server = http.createServer((req, res) => {
		const urlPath = req.url.split("?")[0];
		if (urlPath === "/favicon.ico") {
			res.writeHead(204);
			res.end();
			return;
		}
		if (urlPath.startsWith("/__bundle/")) {
			const relative = urlPath.slice("/__bundle/".length);
			return sendFile(res, path.join(bundleRoot, relative));
		}
		if (urlPath === "/__camera.json") {
			if (!resolvedCameraPath) {
				res.writeHead(404);
				res.end();
				return;
			}
			return sendFile(res, resolvedCameraPath);
		}
		const relativePath = urlPath === "/" ? "/headless.html" : urlPath;
		return sendFile(res, path.join(rootDir, relativePath));
	});
	await new Promise((resolve, reject) => {
		server.once("error", reject);
		server.listen(port, () => {
			server.off("error", reject);
			resolve();
		});
	});
	const address = server.address();
	if (!address || typeof address === "string") {
		throw new Error("Failed to determine headless server port.");
	}
	return { server, port: address.port };
}

async function run() {
	const args = parseArgs(process.argv);
	const rootDir = path.resolve(__dirname);
	const { server, port } = await startServer(
		rootDir,
		args.bundleDir,
		args.camera,
		args.port,
	);
	const browser = await puppeteer.launch({
		headless: "new",
		args: ["--enable-unsafe-webgpu", "--no-sandbox"],
	});
	try {
		const page = await browser.newPage();
		page.on("console", (message) => console.log(`[page] ${message.text()}`));
		page.on("pageerror", (error) =>
			console.error(`[page error] ${error.message}`),
		);
		await page.setViewport({ width: args.width, height: args.height });
		const query = new URLSearchParams({
			bundleBase: "/__bundle",
			time: String(args.time),
			width: String(args.width),
			height: String(args.height),
		});
		if (args.camera) {
			query.set("cameraUrl", "/__camera.json");
		}
		await page.goto(`http://127.0.0.1:${port}/headless.html?${query.toString()}`, {
			waitUntil: "networkidle0",
		});
		await page.waitForFunction(
			() => Boolean(window.__headlessRender && window.__headlessRender.ready),
			{ timeout: 120000 },
		);
		const renderInfo = await page.evaluate(() => window.__headlessRender);
		const canvas = await page.$("#renderCanvas");
		if (!canvas) {
			throw new Error("Missing #renderCanvas.");
		}
		fs.mkdirSync(path.dirname(args.out), { recursive: true });
		await canvas.screenshot({ path: args.out });
		console.log(
			`Rendered ${renderInfo.count} splats at t=${renderInfo.time} -> ${args.out}`,
		);
	} finally {
		await browser.close();
		server.close();
	}
}

run().catch((error) => {
	console.error(error);
	process.exit(1);
});
