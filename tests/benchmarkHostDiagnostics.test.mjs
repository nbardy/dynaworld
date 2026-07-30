import assert from "node:assert/strict";
import test from "node:test";
import {
	cpuBusyFraction,
	evaluateHostSnapshot,
	parseAppleGpuSnapshot,
	parseMemoryPressure,
	parseProcessSnapshot,
	parseSwapUsage,
	parseTopCpuBusy,
} from "../benchmarkHostDiagnostics.js";

test("process snapshots retain only sanitized names and aggregate-ready values", () => {
	const processes = parseProcessSnapshot(`
  123     1  87.5  0.1 /Users/example/Library/Application Support/Steam/steam_osx
  456     1  69.5  3.6 /Applications/ChatGPT.app/Helpers/Codex (Renderer)
  789   456   2.0  0.2 /usr/local/bin/python3
`, new Set([789]));
	assert.deepEqual(processes, [
		{
			pid: 123,
			parentPid: 1,
			name: "steam_osx",
			category: "game-client",
			cpuPercent: 87.5,
			memoryPercent: 0.1,
		},
		{
			pid: 456,
			parentPid: 1,
			name: "Codex (Renderer)",
			category: "developer-tool",
			cpuPercent: 69.5,
			memoryPercent: 3.6,
		},
	]);
	assert.ok(processes.every((process) => !process.name.includes("/")));
});

test("CPU busy fraction uses deltas across all scheduler counters", () => {
	const before = [{
		times: { user: 100, nice: 0, sys: 50, idle: 850, irq: 0 },
	}];
	const after = [{
		times: { user: 200, nice: 0, sys: 100, idle: 1700, irq: 0 },
	}];
	assert.ok(Math.abs(cpuBusyFraction(before, after) - 0.15) < 1e-12);
});

test("macOS top parser uses the latest sampled CPU interval", () => {
	assert.equal(parseTopCpuBusy(`
CPU usage: 19.32% user, 12.97% sys, 67.70% idle
CPU usage: 13.92% user, 6.44% sys, 79.63% idle
`), 0.2036);
});

test("Apple GPU snapshot attributes the latest submitter without persisting its PID", () => {
	const processes = [{
		pid: 26078,
		name: "mediaanalysisd",
		category: "media",
	}];
	const gpu = parseAppleGpuSnapshot(`
"AGCInfo" = {"fLastSubmissionPID"=26078}
"PerformanceStatistics" = {"Alloc system memory"=6473908224,"Tiler Utilization %"=91,"Renderer Utilization %"=92,"Device Utilization %"=94,"In use system memory"=1085145088}
`, processes);
	assert.equal(gpu.available, true);
	assert.equal(gpu.deviceUtilizationPercent, 94);
	assert.equal(gpu.rendererUtilizationPercent, 92);
	assert.deepEqual(gpu.lastSubmissionProcess, {
		name: "mediaanalysisd",
		category: "media",
	});
	assert.equal("pid" in gpu.lastSubmissionProcess, false);
});

test("memory and swap parsers preserve fractions in canonical units", () => {
	assert.equal(parseMemoryPressure("System-wide memory free percentage: 51%"), 0.51);
	const swap = parseSwapUsage(
		"total = 10240.00M  used = 9680.88M  free = 559.12M  (encrypted)",
	);
	assert.equal(swap.totalBytes, 10240 * 1024 ** 2);
	assert.ok(swap.usedFraction > 0.94 && swap.usedFraction < 0.95);
});

test("host validity explains every exceeded contention threshold", () => {
	const assessment = evaluateHostSnapshot({
		platform: "darwin",
		cpuBusyFraction: 0.9,
		cpuBusySource: "top-second-sample",
		loadPerLogicalCpu: 0.8,
		availableMemoryFraction: 0.05,
		processPressure: { competingCpuFraction: 0.5 },
		appleGpu: { available: true, deviceUtilizationPercent: 94 },
		thermal: { thermalWarning: false, performanceWarning: false },
	}, {
		maxCpuBusyFraction: 0.85,
		maxLoadPerLogicalCpu: 0.75,
		maxCompetingCpuFraction: 0.35,
		maxPreflightGpuUtilizationPercent: 35,
		minAvailableMemoryFraction: 0.10,
	});
	assert.equal(assessment.quiet, false);
	assert.equal(assessment.warnings.length, 5);
	assert.match(assessment.warnings.join(" "), /Apple GPU utilization 94%/);
});

test("missing Apple GPU telemetry fails closed for promotion on macOS", () => {
	const assessment = evaluateHostSnapshot({
		platform: "darwin",
		cpuBusyFraction: 0.1,
		cpuBusySource: "top-second-sample",
		loadPerLogicalCpu: 0.1,
		availableMemoryFraction: 0.5,
		processPressure: { competingCpuFraction: 0.1 },
		appleGpu: { available: false },
		thermal: { thermalWarning: false, performanceWarning: false },
	});
	assert.equal(assessment.quiet, false);
	assert.match(assessment.warnings[0], /cannot be proven/);
});

test("missing CPU or process telemetry also fails closed", () => {
	const assessment = evaluateHostSnapshot({
		platform: "darwin",
		cpuBusyFraction: null,
		cpuBusySource: "os.cpus-delta",
		loadPerLogicalCpu: 0.1,
		availableMemoryFraction: 0.5,
		processPressure: { available: false },
		appleGpu: { available: true, deviceUtilizationPercent: 0 },
		thermal: { thermalWarning: false, performanceWarning: false },
	});
	assert.equal(assessment.quiet, false);
	assert.match(assessment.warnings.join(" "), /CPU busy sampling is unavailable/);
	assert.match(assessment.warnings.join(" "), /Process-pressure sampling is unavailable/);
});

test("non-finite process pressure fails closed even when marked available", () => {
	const assessment = evaluateHostSnapshot({
		platform: "darwin",
		cpuBusyFraction: 0.1,
		cpuBusySource: "top-second-sample",
		loadPerLogicalCpu: 0.1,
		availableMemoryFraction: 0.5,
		processPressure: { available: true, competingCpuFraction: null },
		appleGpu: { available: true, deviceUtilizationPercent: 0 },
		thermal: { thermalWarning: false, performanceWarning: false },
	});
	assert.equal(assessment.quiet, false);
	assert.match(assessment.warnings.join(" "), /Process-pressure sampling is unavailable/);
});

test("macOS scheduler-counter fallback cannot promote a benchmark", () => {
	const assessment = evaluateHostSnapshot({
		platform: "darwin",
		cpuBusyFraction: 0,
		cpuBusySource: "os.cpus-delta",
		loadPerLogicalCpu: 0.1,
		availableMemoryFraction: 0.5,
		processPressure: { available: true, competingCpuFraction: 0.1 },
		appleGpu: { available: true, deviceUtilizationPercent: 0 },
		thermal: { thermalWarning: false, performanceWarning: false },
	});
	assert.equal(assessment.quiet, false);
	assert.match(assessment.warnings.join(" "), /did not use top's second interval/);
});
