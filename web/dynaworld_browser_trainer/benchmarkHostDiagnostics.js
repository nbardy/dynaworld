import { execFile } from "node:child_process";
import os from "node:os";
import path from "node:path";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);

const DEFAULT_THRESHOLDS = Object.freeze({
	maxCpuBusyFraction: 0.85,
	maxLoadPerLogicalCpu: 0.75,
	maxCompetingCpuFraction: 0.35,
	maxPreflightGpuUtilizationPercent: 35,
	minAvailableMemoryFraction: 0.10,
});

function round(value, digits = 4) {
	if (!Number.isFinite(value)) return null;
	const scale = 10 ** digits;
	return Math.round(value * scale) / scale;
}

function basename(command) {
	return path.basename(command.trim()).replace(/\s+/g, " ").slice(0, 80);
}

export function categorizeProcess(name) {
	const normalized = name.toLowerCase();
	if (/(chrome|chromium|firefox|safari)/.test(normalized)) return "browser";
	if (/(steam|battle\.net|epic games)/.test(normalized)) return "game-client";
	if (/(mediaanalysis|webtorrent|vtdecoder|ffmpeg)/.test(normalized)) return "media";
	if (/(codex|chatgpt|claude)/.test(normalized)) return "developer-tool";
	if (/(python|node|bun|deno)/.test(normalized)) return "runtime";
	if (/(windowserver)/.test(normalized)) return "system-ui";
	return "other";
}

export function parseProcessSnapshot(text, excludedPids = new Set()) {
	return text
		.split("\n")
		.map((line) => {
			const match = line.match(
				/^\s*(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(.+?)\s*$/,
			);
			if (!match) return null;
			const pid = Number(match[1]);
			if (excludedPids.has(pid)) return null;
			const name = basename(match[5]);
			return {
				pid,
				parentPid: Number(match[2]),
				name,
				category: categorizeProcess(name),
				cpuPercent: Number(match[3]),
				memoryPercent: Number(match[4]),
			};
		})
		.filter(Boolean);
}

export function cpuBusyFraction(beforeCpus, afterCpus) {
	const totals = (cpus) => cpus.reduce((result, cpu) => {
		for (const [key, value] of Object.entries(cpu.times)) result[key] += value;
		return result;
	}, { user: 0, nice: 0, sys: 0, idle: 0, irq: 0 });
	const before = totals(beforeCpus);
	const after = totals(afterCpus);
	const elapsed = Object.keys(before).reduce(
		(sum, key) => sum + Math.max(0, after[key] - before[key]),
		0,
	);
	const idle = Math.max(0, after.idle - before.idle);
	return elapsed > 0 ? 1 - idle / elapsed : Number.NaN;
}

export function parseMemoryPressure(text) {
	const match = text.match(/free percentage:\s*(\d+(?:\.\d+)?)%/i);
	return match ? Number(match[1]) / 100 : null;
}

export function parseSwapUsage(text) {
	const values = Object.fromEntries(
		[...text.matchAll(/(total|used|free)\s*=\s*([\d.]+)([KMG])/gi)].map((match) => {
			const multiplier = { K: 1024, M: 1024 ** 2, G: 1024 ** 3 }[match[3].toUpperCase()];
			return [match[1].toLowerCase(), Number(match[2]) * multiplier];
		}),
	);
	if (!Number.isFinite(values.total) || !Number.isFinite(values.used)) return null;
	return {
		totalBytes: Math.round(values.total),
		usedBytes: Math.round(values.used),
		usedFraction: values.total > 0 ? values.used / values.total : null,
	};
}

export function parseTopCpuBusy(text) {
	const samples = [...text.matchAll(
		/CPU usage:\s*([\d.]+)% user,\s*([\d.]+)% sys,\s*([\d.]+)% idle/gi,
	)];
	if (!samples.length) return null;
	const latest = samples.at(-1);
	return (Number(latest[1]) + Number(latest[2])) / 100;
}

export function parseAppleGpuSnapshot(text, processes = []) {
	const utilization = (label) => {
		const match = text.match(new RegExp(`"${label} Utilization %"=(\\d+)`));
		return match ? Number(match[1]) : null;
	};
	const lastPidMatch = text.match(/"fLastSubmissionPID"=(\d+)/);
	const lastProcess = lastPidMatch
		? processes.find((process) => process.pid === Number(lastPidMatch[1]))
		: null;
	const memory = (label) => {
		const match = text.match(new RegExp(`"${label}"=(\\d+)`));
		return match ? Number(match[1]) : null;
	};
	const deviceUtilizationPercent = utilization("Device");
	if (deviceUtilizationPercent == null) {
		return { available: false, reason: "Apple GPU utilization was not reported." };
	}
	return {
		available: true,
		deviceUtilizationPercent,
		rendererUtilizationPercent: utilization("Renderer"),
		tilerUtilizationPercent: utilization("Tiler"),
		allocatedSystemMemoryBytes: memory("Alloc system memory"),
		inUseSystemMemoryBytes: memory("In use system memory"),
		lastSubmissionProcess: lastProcess ? {
			name: lastProcess.name,
			category: lastProcess.category,
		} : null,
	};
}

export function parseThermalStatus(text) {
	return {
		thermalWarning: !/No thermal warning level has been recorded/i.test(text),
		performanceWarning: !/No performance warning level has been recorded/i.test(text),
	};
}

async function optionalCommand(command, args) {
	try {
		const { stdout } = await execFileAsync(command, args, {
			maxBuffer: 5 * 1024 * 1024,
			timeout: 5000,
		});
		return { available: true, stdout };
	} catch (error) {
		return {
			available: false,
			reason: `${path.basename(command)} unavailable (${error.code ?? "command failed"})`,
		};
	}
}

function delay(milliseconds) {
	return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

function summarizeProcesses(processes, logicalCpuCount) {
	const active = processes
		.filter((process) => process.cpuPercent >= 2)
		.sort((left, right) => right.cpuPercent - left.cpuPercent);
	const competingCpuPercent = active.reduce(
		(sum, process) => sum + process.cpuPercent,
		0,
	);
	const categoryCpuPercent = {};
	for (const process of active) {
		categoryCpuPercent[process.category] =
			(categoryCpuPercent[process.category] ?? 0) + process.cpuPercent;
	}
	return {
		available: true,
		activeProcessCount: active.length,
		competingCpuPercent: round(competingCpuPercent, 2),
		competingCpuFraction: round(
			competingCpuPercent / (100 * Math.max(logicalCpuCount, 1)),
		),
		categoryCpuPercent: Object.fromEntries(
			Object.entries(categoryCpuPercent)
				.sort(([, left], [, right]) => right - left)
				.map(([category, value]) => [category, round(value, 2)]),
		),
		// Names are basenames only: no arguments, paths, or PIDs enter artifacts.
		topProcesses: active.slice(0, 12).map((process) => ({
			name: process.name,
			category: process.category,
			cpuPercent: process.cpuPercent,
			memoryPercent: process.memoryPercent,
		})),
	};
}

export function evaluateHostSnapshot(snapshot, thresholds = DEFAULT_THRESHOLDS) {
	const warnings = [];
	if (!Number.isFinite(snapshot.cpuBusyFraction)) {
		warnings.push("CPU busy sampling is unavailable; a quiet host cannot be proven.");
	} else if (
		snapshot.platform === "darwin"
		&& snapshot.cpuBusySource !== "top-second-sample"
	) {
		warnings.push(
			"macOS CPU busy sampling did not use top's second interval; "
			+ "a quiet host cannot be proven.",
		);
	} else if (snapshot.cpuBusyFraction > thresholds.maxCpuBusyFraction) {
		warnings.push(
			`CPU busy fraction ${round(snapshot.cpuBusyFraction, 3)} exceeds `
			+ `${thresholds.maxCpuBusyFraction}.`,
		);
	}
	if (snapshot.loadPerLogicalCpu > thresholds.maxLoadPerLogicalCpu) {
		warnings.push(
			`1-minute load per logical CPU ${round(snapshot.loadPerLogicalCpu, 3)} exceeds `
			+ `${thresholds.maxLoadPerLogicalCpu}.`,
		);
	}
	if (
		snapshot.processPressure.available === false
		|| !Number.isFinite(snapshot.processPressure.competingCpuFraction)
	) {
		warnings.push("Process-pressure sampling is unavailable; a quiet host cannot be proven.");
	} else if (
		snapshot.processPressure.competingCpuFraction
		> thresholds.maxCompetingCpuFraction
	) {
		warnings.push(
			`Competing process CPU fraction ${snapshot.processPressure.competingCpuFraction} `
			+ `exceeds ${thresholds.maxCompetingCpuFraction}.`,
		);
	}
	if (
		snapshot.appleGpu.available
		&& snapshot.appleGpu.deviceUtilizationPercent
			> thresholds.maxPreflightGpuUtilizationPercent
	) {
		warnings.push(
			`Apple GPU utilization ${snapshot.appleGpu.deviceUtilizationPercent}% `
			+ `exceeds ${thresholds.maxPreflightGpuUtilizationPercent}%.`,
		);
	} else if (snapshot.platform === "darwin" && !snapshot.appleGpu.available) {
		warnings.push(
			"Apple GPU utilization is unavailable; a quiet GPU preflight cannot be proven.",
		);
	}
	if (
		Number.isFinite(snapshot.availableMemoryFraction)
		&& snapshot.availableMemoryFraction < thresholds.minAvailableMemoryFraction
	) {
		warnings.push(
			`Available memory fraction ${round(snapshot.availableMemoryFraction, 3)} is below `
			+ `${thresholds.minAvailableMemoryFraction}.`,
		);
	}
	if (snapshot.thermal.thermalWarning || snapshot.thermal.performanceWarning) {
		warnings.push("macOS reports thermal or performance pressure.");
	}
	return {
		quiet: warnings.length === 0,
		warnings,
		thresholds,
	};
}

export async function captureHostSnapshot({
	sampleMs = 750,
	thresholds = DEFAULT_THRESHOLDS,
	excludedPids = new Set([process.pid]),
} = {}) {
	const logicalCpuCount = os.cpus().length;
	const cpuBefore = os.cpus();
	const commands = process.platform === "darwin" ? {
		processes: optionalCommand(
			"/bin/ps",
			["-axo", "pid=,ppid=,pcpu=,pmem=,comm="],
		),
		memoryPressure: optionalCommand("/usr/bin/memory_pressure", ["-Q"]),
		swap: optionalCommand("/usr/sbin/sysctl", ["-n", "vm.swapusage"]),
		gpu: optionalCommand(
			"/usr/sbin/ioreg",
			["-r", "-d", "1", "-w", "0", "-c", "AGXAccelerator"],
		),
		thermal: optionalCommand("/usr/bin/pmset", ["-g", "therm"]),
		cpu: optionalCommand("/usr/bin/top", ["-l", "2", "-s", "1", "-n", "0"]),
	} : {
		processes: optionalCommand(
			"/bin/ps",
			["-axo", "pid=,ppid=,pcpu=,pmem=,comm="],
		),
	};
	await delay(sampleMs);
	const cpuAfter = os.cpus();
	const commandResults = Object.fromEntries(
		await Promise.all(
			Object.entries(commands).map(async ([name, promise]) => [name, await promise]),
		),
	);
	const processTelemetryAvailable = commandResults.processes?.available === true;
	const processes = processTelemetryAvailable
		? parseProcessSnapshot(commandResults.processes.stdout, excludedPids)
		: [];
	const availableMemoryFraction = commandResults.memoryPressure?.available
		? parseMemoryPressure(commandResults.memoryPressure.stdout)
		: os.freemem() / os.totalmem();
	const appleGpu = commandResults.gpu?.available
		? parseAppleGpuSnapshot(commandResults.gpu.stdout, processes)
		: {
			available: false,
			reason: commandResults.gpu?.reason ?? "Apple GPU diagnostics are not available.",
		};
	const thermal = commandResults.thermal?.available
		? parseThermalStatus(commandResults.thermal.stdout)
		: { thermalWarning: false, performanceWarning: false, available: false };
	const loadAverage = os.loadavg();
	const externalCpuBusy = commandResults.cpu?.available
		? parseTopCpuBusy(commandResults.cpu.stdout) : null;
	const schedulerCpuBusy = cpuBusyFraction(cpuBefore, cpuAfter);
	const sampledCpuBusy = Number.isFinite(externalCpuBusy)
		? externalCpuBusy : schedulerCpuBusy;
	const snapshot = {
		schema: "dynaworld-benchmark-host-snapshot/v1",
		recordedAt: new Date().toISOString(),
		platform: process.platform,
		architecture: process.arch,
		sampleMs,
		logicalCpuCount,
		loadAverage1m: round(loadAverage[0]),
		loadAverage5m: round(loadAverage[1]),
		loadAverage15m: round(loadAverage[2]),
		loadPerLogicalCpu: round(loadAverage[0] / Math.max(logicalCpuCount, 1)),
		cpuBusyFraction: round(sampledCpuBusy),
		cpuBusySource: Number.isFinite(externalCpuBusy)
			? "top-second-sample" : "os.cpus-delta",
		totalMemoryBytes: os.totalmem(),
		availableMemoryFraction: round(availableMemoryFraction),
		memoryAvailabilitySource: commandResults.memoryPressure?.available
			? "memory_pressure" : "os.freemem",
		swap: commandResults.swap?.available
			? parseSwapUsage(commandResults.swap.stdout) : null,
		processPressure: processTelemetryAvailable
			? summarizeProcesses(processes, logicalCpuCount)
			: {
				available: false,
				reason: commandResults.processes?.reason ?? "Process telemetry is unavailable.",
			},
		appleGpu,
		thermal,
		limitations: [
			"Process names are sanitized basenames; command arguments and PIDs are omitted.",
			"Driver GPU utilization is a pre/post snapshot, not per-process GPU accounting.",
			"Round variance remains the in-run detector for nonlinear contention and throttling.",
		],
	};
	snapshot.assessment = evaluateHostSnapshot(snapshot, thresholds);
	return snapshot;
}

export { DEFAULT_THRESHOLDS };
