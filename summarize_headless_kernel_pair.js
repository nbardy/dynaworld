#!/usr/bin/env bun

import fs from "node:fs";
import path from "node:path";
import { summarizeBenchmarkPair } from "./benchmarkPairStatistics.js";

function usage() {
	return `Usage: bun web/dynaworld_browser_trainer/summarize_headless_kernel_pair.js [options] RUN_A RUN_B

Validates a reversed-start pair of v3 headless WebGPU benchmark artifacts.

  --out PATH                       write pair summary JSON (default: stdout)
  --max-wall-speedup-drift N       default: 0.05
  --max-gpu-speedup-drift N        default: 0.10
  --max-variant-throughput-drift N default: 0.05
`;
}

function parseArgs(argv) {
	const options = {
		out: null,
		maxWallSpeedupDrift: 0.05,
		maxGpuSpeedupDrift: 0.10,
		maxVariantThroughputDrift: 0.05,
		files: [],
	};
	for (let index = 2; index < argv.length; index += 1) {
		const value = argv[index];
		if (value === "--help" || value === "-h") return { help: true };
		if (!value.startsWith("-")) {
			options.files.push(value);
			continue;
		}
		const raw = argv[++index];
		if (raw === undefined) throw new Error(`${value} requires a value.`);
		if (value === "--out") options.out = raw;
		else if (value === "--max-wall-speedup-drift") {
			options.maxWallSpeedupDrift = Number(raw);
		} else if (value === "--max-gpu-speedup-drift") {
			options.maxGpuSpeedupDrift = Number(raw);
		} else if (value === "--max-variant-throughput-drift") {
			options.maxVariantThroughputDrift = Number(raw);
		} else {
			throw new Error(`Unknown option: ${value}`);
		}
	}
	if (options.files.length !== 2) throw new Error("Exactly two run artifacts are required.");
	for (const key of [
		"maxWallSpeedupDrift",
		"maxGpuSpeedupDrift",
		"maxVariantThroughputDrift",
	]) {
		if (!Number.isFinite(options[key]) || options[key] < 0 || options[key] > 1) {
			throw new Error(`${key} must be from 0 through 1.`);
		}
	}
	return options;
}

function readReport(file) {
	const report = JSON.parse(fs.readFileSync(file, "utf8"));
	if (report.schema !== "dynaworld-browser-tiled-kernel-benchmark/v3") {
		throw new Error(`${file} is not a v3 tiled-kernel artifact.`);
	}
	return report;
}

const options = parseArgs(process.argv);
if (options.help) {
	process.stdout.write(usage());
	process.exit(0);
}
const summary = summarizeBenchmarkPair(
	readReport(options.files[0]),
	readReport(options.files[1]),
	options,
);
summary.sources = options.files.map((file) => path.basename(file));
const json = `${JSON.stringify(summary, null, 2)}\n`;
if (options.out) {
	const outputPath = path.resolve(options.out);
	fs.mkdirSync(path.dirname(outputPath), { recursive: true });
	fs.writeFileSync(outputPath, json);
	process.stderr.write(
		`Wrote ${summary.validity.promotable ? "promotable" : "diagnostic-only"} `
		+ `pair summary to ${outputPath}\n`,
	);
} else {
	process.stdout.write(json);
}
if (!summary.validity.promotable) process.exitCode = 2;
