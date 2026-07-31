import { drawTargetFrame, loadPresetDataset } from "./dataset.js?v=20260731-compactfp16-5";
import { createNonblockingTrainer } from "./nonblockingTrainerClient.js?v=20260731-compactfp16-5";
import { learningRateMultipliers } from "./trainingSchedule.js?v=20260731-compactfp16-5";
import { DynamicSplatWebGpuTrainer } from "./trainerWebGpu.js?v=20260731-compactfp16-5";
import { StatusFlag, WorkerEvent } from "./workerProtocol.js?v=20260731-compactfp16-5";

const $ = (id) => document.getElementById(id);
const renderCanvas = $("renderCanvas");
const targetCanvas = $("targetCanvas");
const comparisonCanvases = [$("sourceViewCanvas"), $("targetViewCanvas"), $("heldoutViewCanvas")];
const comparisonFrameLabels = [$("sourceViewFrameValue"), $("targetViewFrameValue"), $("heldoutViewFrameValue")];
const resultViewLabels = [0, 1, 2].map((index) => $(`resultViewLabel${index}`));
const resultViewRoles = [0, 1, 2].map((index) => $(`resultViewRole${index}`));
const controls = {
	run: $("runButton"), step: $("stepButton"), reset: $("resetButton"), backend: $("backendSelect"),
	precision: $("checkpointPrecisionSelect"), mode: $("modeSelect"),
	splats: $("splatSlider"), growthCapacity: $("growthCapacitySelect"),
	time: $("timeSlider"), loop: $("timeLoopToggle"), live: $("livePreviewToggle"),
	fullMetrics: $("fullMetricsToggle"), speed: $("timeSpeedSlider"), targetView: $("targetViewSelect"),
	renderCamera: $("renderCameraSelect"), resultView: $("resultViewSelect"),
	temporalSchedule: $("temporalScheduleToggle"), temporal: $("temporalSlider"), lr: $("lrSlider"),
	lrSchedule: $("lrScheduleToggle"), staticWarmup: $("staticWarmupToggle"),
	motionWeighting: $("motionWeightingToggle"),
	randomBackground: $("randomBackgroundToggle"),
	samples: $("samplesSlider"), motionMix: $("motionMixSlider"), staticMix: $("staticMixSlider"),
	supportGuard: $("supportGuardSlider"),
};
const values = {
	splats: $("splatSliderValue"), time: $("timeValue"), speed: $("timeSpeedValue"),
	temporalLabel: $("temporalSliderLabel"), temporal: $("temporalValue"),
	temporalSchedule: $("temporalScheduleValue"), lr: $("lrValue"), lrSchedule: $("lrScheduleValue"),
	samples: $("samplesValue"),
	motionMix: $("motionMixValue"), staticMix: $("staticMixValue"), supportGuard: $("supportGuardValue"),
	step: $("stepValue"), phase: $("phaseValue"), stepRate: $("stepRateValue"), sampleLoss: $("lossValue"),
	gridLoss: $("gridLossValue"), trainMae: $("valMaeValue"), trainPsnr: $("valPsnrValue"),
	trainSsim: $("valSsimValue"), heldoutLoss: $("heldoutLossValue"), heldoutMae: $("heldoutMaeValue"),
	heldoutPsnr: $("heldoutPsnrValue"), heldoutSsim: $("heldoutSsimValue"),
	heldoutCoverage: $("heldoutCoverageValue"), motionLoss: $("motionLossValue"),
	motionCoverage: $("motionCoverageValue"), staticCoverage: $("staticCoverageValue"),
	peakAlpha: $("motionMaxAlphaValue"), active: $("activeSplatValue"), meanOpacity: $("meanOpacityValue"),
	meanRadius: $("meanRadiusValue"), meanAspect: $("meanAspectValue"), tileOverflow: $("tileOverflowValue"),
	metricPair: $("metricPairValue"), visibleSplats: $("visibleSplatValue"),
	tilePairs: $("tilePairValue"), tileLoad: $("tileLoadValue"),
	detailMae: $("detailMaeValue"), lowPassPsnr: $("lowPassPsnrValue"),
	cameraPsnr: $("cameraPsnrValue"), gpuMemory: $("gpuMemoryValue"),
	representation: $("representationValue"), dynamicSplats: $("dynamicSplatValue"),
	persistentSplats: $("persistentSplatValue"), staticMixP50: $("staticMixP50Value"),
	edgeSupport: $("edgeSupportValue"), aspectP90: $("aspectP90Value"),
	rasterDead: $("rasterDeadValue"),
	splatCount: $("splatValue"), recycled: $("recycledSplatValue"),
	parameterDelta: $("parameterDeltaValue"), motionSamples: $("motionSampleValue"),
	centerUpdate: $("centerUpdateValue"), motionUpdate: $("motionUpdateValue"),
	scaleUpdate: $("scaleUpdateValue"), rotationUpdate: $("rotationUpdateValue"),
	colorUpdate: $("colorUpdateValue"), opacityUpdate: $("opacityUpdateValue"),
	staticSamples: $("staticSampleValue"), gpu: $("gpuValue"), fps: $("fpsValue"),
	runtime: $("runtimeValue"), shared: $("sharedStatusValue"), validation: $("validationRuntimeValue"),
	seedProvenance: $("seedProvenanceValue"),
};
const chartElements = {
	loss: $("lossChartCanvas"), lossRange: $("lossChartRange"), lossScale: $("lossScaleToggle"),
	psnr: $("psnrChartCanvas"), psnrRange: $("psnrChartRange"),
	ssim: $("ssimChartCanvas"), ssimRange: $("ssimChartRange"),
};

const TEMPORAL_SUPPORT_START = 0.30;
const TEMPORAL_SUPPORT_TARGET = 0.26;
const TEMPORAL_SUPPORT_HOLD_STEPS = 256;
const TEMPORAL_SUPPORT_END_STEP = 2048;
const RENDER_FPS = 15;
const MAX_RENDER_WIDTH = 960;
const VALIDATION_STEP_INTERVAL = 8192;
const STATIC_WARMUP_STEPS = 2048;
const UI_STATE_KEY = "dynaworld-browser-trainer-ui-v2";
const metricHistory = { sampleLoss: [], trainLoss: [], heldoutLoss: [],
	trainPsnr: [], heldoutPsnr: [], trainSsim: [], heldoutSsim: [] };

let dataset = null;
let workerClient = null;
let localTrainer = null;
let running = false;
let booting = true;
let lossEma = null;
let lossLogScale = true;
let animationHandle = 0;
let lastFrameAt = performance.now();
let lastTargetDrawAt = 0;
let lastRenderOptionsKey = "";
let lastTrainOptionsKey = "";
let lastStatusMetricStep = -1;
let lastValidationRequestStep = 0;
let localLossPending = false;
let lastLocalLossStep = -1;
let trainerCapacity = 0;
let trainerStaticWarmupSteps = 0;
const frameDurations = [];

function currentStep() {
	return workerClient?.getStatus()?.step ?? localTrainer?.stepCount ?? 0;
}

function currentModelMode() {
	return controls.mode.value === "dynamic_splats" ? 1 : 0;
}

function currentRenderMode() {
	if (controls.resultView.value === "alpha_support") return 2;
	return controls.resultView.value === "dynamic_residual" ? 1 : 0;
}

function currentTemporalSigma(step = currentStep()) {
	if (!controls.temporalSchedule.checked) return Number(controls.temporal.value);
	const progress = Math.max(0, Math.min(1, (step - TEMPORAL_SUPPORT_HOLD_STEPS)
		/ (TEMPORAL_SUPPORT_END_STEP - TEMPORAL_SUPPORT_HOLD_STEPS)));
	const smooth = progress * progress * (3 - 2 * progress);
	return TEMPORAL_SUPPORT_START + (TEMPORAL_SUPPORT_TARGET - TEMPORAL_SUPPORT_START) * smooth;
}

function effectiveMotionMix() {
	return Math.min(Number(controls.motionMix.value), 1 - Number(controls.staticMix.value));
}

function sampledBackendSelected() {
	return controls.backend.value === "sampled3d";
}

function trainOptions() {
	return {
		learningRate: Number(controls.lr.value), learningRateDecay: controls.lrSchedule.checked,
		samplesPerStep: Number(controls.samples.value),
		modelMode: currentModelMode(), temporalSigma: currentTemporalSigma(),
		motionSampleRate: effectiveMotionMix(), staticSampleRate: Number(controls.staticMix.value),
		motionCoverageTarget: Number(controls.supportGuard.value),
		motionWeighting: controls.motionWeighting.checked,
		randomBackground: !sampledBackendSelected() && controls.randomBackground.checked,
		camerasPerStep: 4,
	};
}

function renderOptions() {
	return { time: Number(controls.time.value), modelMode: currentModelMode(),
		temporalSigma: currentTemporalSigma(), renderMode: currentRenderMode(),
		enabled: controls.live.checked,
		viewIndex: Number(controls.renderCamera.value || 0),
		viewIndices: dataset?.comparisonViewIndices ?? null };
}

function setStatus(message) {
	$("statusText").textContent = message;
}

function setRunning(next) {
	running = Boolean(next);
	controls.run.textContent = running ? "Pause" : "Start";
	controls.run.dataset.running = String(running);
	controls.step.disabled = running || booting;
	if (workerClient) {
		if (running) workerClient.start(); else workerClient.pause();
	}
}

function updateControlLabels() {
	const splatLimit = sampledBackendSelected() ? 2048 : 4096;
	controls.splats.max = String(splatLimit);
	if (Number(controls.splats.value) > splatLimit) controls.splats.value = String(splatLimit);
	const step = currentStep();
	const sigma = currentTemporalSigma(step);
	const lrMultipliers = learningRateMultipliers(step, controls.lrSchedule.checked);
	const baseLearningRate = Number(controls.lr.value);
	const formatLearningRate = (value) => value >= 0.1 ? value.toFixed(2) : value.toPrecision(2);
	values.splats.textContent = controls.splats.value;
	values.time.textContent = Number(controls.time.value).toFixed(3);
	values.speed.textContent = `${Number(controls.speed.value).toFixed(2)}x`;
	values.temporal.textContent = sigma.toFixed(3);
	values.lr.textContent = `${baseLearningRate.toFixed(2)}x`;
	values.lrSchedule.textContent = controls.lrSchedule.checked
		? `geometry ${formatLearningRate(baseLearningRate * lrMultipliers.geometry)}x · `
			+ `appearance ${formatLearningRate(baseLearningRate * lrMultipliers.appearance)}x`
		: "fixed legacy control";
	values.samples.textContent = sampledBackendSelected()
		? controls.samples.value : dataset ? `${dataset.width * dataset.height} px` : "full image";
	values.motionMix.textContent = `${Math.round(effectiveMotionMix() * 100)}%`;
	values.staticMix.textContent = `${Math.round(Number(controls.staticMix.value) * 100)}%`;
	values.supportGuard.textContent = `${Math.round(Number(controls.supportGuard.value) * 100)}%`;
	controls.temporal.disabled = controls.temporalSchedule.checked;
	for (const control of [controls.samples, controls.motionMix, controls.staticMix, controls.supportGuard]) {
		control.disabled = !sampledBackendSelected();
	}
	for (const field of [$("sampleCountField"), $("motionMixField"), $("staticMixField"), $("supportGuardField")]) {
		field.toggleAttribute("data-disabled", !sampledBackendSelected());
	}
	controls.precision.disabled = sampledBackendSelected();
	$("checkpointPrecisionField").toggleAttribute("data-disabled", sampledBackendSelected());
	controls.growthCapacity.disabled = sampledBackendSelected();
	$("growthCapacityField").toggleAttribute("data-disabled", sampledBackendSelected());
	controls.staticWarmup.disabled = sampledBackendSelected();
	$("staticWarmupField").toggleAttribute("data-disabled", sampledBackendSelected());
	controls.motionWeighting.disabled = sampledBackendSelected();
	$("motionWeightingField").toggleAttribute("data-disabled", sampledBackendSelected());
	controls.randomBackground.disabled = sampledBackendSelected();
	$("randomBackgroundField").toggleAttribute("data-disabled", sampledBackendSelected());
	values.temporalLabel.textContent = controls.temporalSchedule.checked ? "Temporal Support Now" : "Temporal Support";
	if (!controls.temporalSchedule.checked) {
		values.temporalSchedule.textContent = "manual · fixed";
	} else if (step < TEMPORAL_SUPPORT_HOLD_STEPS) {
		values.temporalSchedule.textContent = `hold ${step}/${TEMPORAL_SUPPORT_HOLD_STEPS} · target 0.26`;
	} else if (step < TEMPORAL_SUPPORT_END_STEP) {
		const progress = Math.round(100 * (step - TEMPORAL_SUPPORT_HOLD_STEPS)
			/ (TEMPORAL_SUPPORT_END_STEP - TEMPORAL_SUPPORT_HOLD_STEPS));
		values.temporalSchedule.textContent = `narrowing ${progress}% · target 0.26`;
	} else {
		values.temporalSchedule.textContent = "settled · target 0.26";
	}
}

function drawMetricChart(canvas, range, definitions, { log = false, format = (value) => value.toFixed(3) } = {}) {
	const ctx = canvas.getContext("2d");
	if (!ctx) return;
	const dpr = window.devicePixelRatio || 1;
	const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
	const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
	if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
	ctx.clearRect(0, 0, width, height);
	ctx.strokeStyle = "#2d353e"; ctx.lineWidth = dpr;
	for (let row = 1; row < 4; row += 1) {
		const y = Math.round((height * row) / 4) + 0.5;
		ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
	}
	const valid = (point) => Number.isFinite(point.value) && (!log || point.value > 0);
	const points = definitions.flatMap(({ key }) => metricHistory[key]).filter(valid);
	if (!points.length) { range.textContent = "waiting"; return; }
	const minStep = Math.min(...points.map((point) => point.step));
	const historyMaxStep = Math.max(...points.map((point) => point.step));
	const maxStep = Math.max(minStep + 1, historyMaxStep);
	const transformed = points.map((point) => log ? Math.log10(point.value) : point.value);
	let minValue = Math.min(...transformed); let maxValue = Math.max(...transformed);
	const minimumSpan = log ? 0.08 : Math.max(0.02, Math.abs(maxValue) * 0.025);
	if (maxValue - minValue < minimumSpan) {
		const center = (minValue + maxValue) / 2; minValue = center - minimumSpan / 2; maxValue = center + minimumSpan / 2;
	}
	const inset = 8 * dpr;
	for (const { key, color } of definitions) {
		const visible = metricHistory[key].filter(valid); if (!visible.length) continue;
		ctx.beginPath(); ctx.strokeStyle = color; ctx.lineWidth = 1.7 * dpr;
		ctx.lineJoin = "round"; ctx.lineCap = "round";
		visible.forEach((point, index) => {
			const x = inset + ((point.step - minStep) / (maxStep - minStep)) * (width - inset * 2);
			const value = log ? Math.log10(point.value) : point.value;
			const y = inset + ((maxValue - value) / (maxValue - minValue)) * (height - inset * 2);
			if (index) ctx.lineTo(x, y); else ctx.moveTo(x, y);
		});
		ctx.stroke();
	}
	const latest = points.reduce((current, point) => point.step >= current.step ? point : current, points[0]);
	range.textContent = `${format(latest.value)} @ ${latest.step} · full ${minStep}–${historyMaxStep}`;
}

function drawMetricCharts() {
	drawMetricChart(chartElements.loss, chartElements.lossRange, [
		{ key: "sampleLoss", color: "#56c7a5" }, { key: "trainLoss", color: "#e1b85f" },
		{ key: "heldoutLoss", color: "#e67c73" },
	], { log: lossLogScale, format: (value) => value.toExponential(2) });
	drawMetricChart(chartElements.psnr, chartElements.psnrRange, [
		{ key: "trainPsnr", color: "#e1b85f" }, { key: "heldoutPsnr", color: "#e67c73" },
	], { format: (value) => `${value.toFixed(1)} dB` });
	drawMetricChart(chartElements.ssim, chartElements.ssimRange, [
		{ key: "trainSsim", color: "#e1b85f" }, { key: "heldoutSsim", color: "#e67c73" },
	], { format: (value) => value.toFixed(3) });
}

function recordMetric(kind, step, value) {
	if (!Number.isFinite(value) || (kind.endsWith("Loss") && value <= 0)) return;
	const series = metricHistory[kind];
	if (series.at(-1)?.step === step) series[series.length - 1] = { step, value };
	else series.push({ step, value });
	drawMetricCharts();
}

function resetMetrics() {
	for (const series of Object.values(metricHistory)) series.length = 0;
	for (const value of [values.sampleLoss, values.gridLoss, values.trainMae, values.trainPsnr,
		values.trainSsim, values.heldoutLoss, values.heldoutMae, values.heldoutPsnr,
		values.heldoutSsim, values.heldoutCoverage, values.motionLoss, values.motionCoverage,
		values.staticCoverage, values.peakAlpha, values.active, values.meanOpacity, values.meanRadius,
		values.meanAspect, values.dynamicSplats, values.persistentSplats, values.staticMixP50,
		values.edgeSupport, values.aspectP90, values.rasterDead, values.tileOverflow,
		values.metricPair, values.visibleSplats, values.tilePairs, values.tileLoad,
		values.detailMae, values.lowPassPsnr, values.cameraPsnr, values.gpuMemory,
		values.parameterDelta, values.centerUpdate,
		values.motionUpdate, values.scaleUpdate, values.rotationUpdate, values.colorUpdate,
		values.opacityUpdate]) {
		value.textContent = "--";
	}
	delete values.tileOverflow.dataset.state;
	drawMetricCharts();
}

function consumeSampleMetric({ step, loss, breakdown = null, totalRecycled = Number.NaN }) {
	if (!Number.isFinite(loss)) {
		values.sampleLoss.textContent = "invalid";
		const detail = breakdown ? ` (${JSON.stringify(breakdown)})` : "";
		setStatus(`Training objective became non-finite at step ${step}${detail}.`);
		return;
	}
	if (loss < 0) return;
	if (loss === 0 && breakdown) {
		setStatus(`Zero-objective diagnostic at step ${step}: ${JSON.stringify(breakdown)}.`);
	}
	const cycleMeanLoss = Number(breakdown?.cycleMeanLoss);
	if (Number.isFinite(cycleMeanLoss)) {
		lossEma = cycleMeanLoss;
		const cycleSamples = Math.round(Number(breakdown.cycleSamples));
		values.sampleLoss.title = `GPU rolling camera/time mean over ${cycleSamples} objective steps; `
			+ `current pair ${loss.toFixed(5)}`;
	} else {
		lossEma = lossEma == null ? loss : lossEma * 0.82 + loss * 0.18;
		values.sampleLoss.title = `Sampled-objective exponential moving average; current sample ${loss.toFixed(5)}`;
	}
	values.sampleLoss.textContent = lossEma.toFixed(5);
	if (breakdown) {
		const overflow = Number(breakdown.tileOverflow);
		const overflowTotal = Number(breakdown.tileOverflowTotal);
		values.tileOverflow.textContent = Number.isFinite(overflow)
			? `${Math.round(overflow)} / ${Number.isFinite(overflowTotal) ? Math.round(overflowTotal) : "?"}`
			: "--";
		values.tileOverflow.dataset.state = overflow > 0 || overflowTotal > 0 ? "warning" : "ok";
		const viewIndex = Math.round(Number(breakdown.viewIndex));
		const frameIndex = Math.round(Number(breakdown.frameIndex));
		const camera = dataset?.cameras?.[viewIndex]?.name ?? `cam${viewIndex}`;
		const phase = Number.isFinite(breakdown.cyclePhase) && Number.isFinite(breakdown.pairCycle)
			? ` · ${Math.round(breakdown.cyclePhase) + 1}/${Math.round(breakdown.pairCycle)}` : "";
		const topology = Number(breakdown.topologyOpsSinceMetric) > 0
			? ` · +${Math.round(breakdown.topologyOpsSinceMetric)} split` : "";
		values.metricPair.textContent = `${camera} f${frameIndex}${phase}${topology}`;
		setMetricText(values.visibleSplats, breakdown.visibleSplats,
			(value) => `${Math.round(value)}/${Math.round(breakdown.capacitySplats)}`);
		setMetricText(values.tilePairs, breakdown.pairCount, (value) => Math.round(value).toLocaleString());
		if (Number.isFinite(breakdown.maxTileOccupancy) && Number.isFinite(breakdown.meanStopRank)) {
			const maximumEver = Number.isFinite(breakdown.maxTileOccupancyEver)
				? ` (${Math.round(breakdown.maxTileOccupancyEver)})` : "";
			values.tileLoad.textContent = `${Math.round(breakdown.maxTileOccupancy)}${maximumEver} / `
				+ `${Number(breakdown.meanStopRank).toFixed(1)}`;
		}
	}
	setMetricText(values.recycled, totalRecycled, (value) => String(value));
	recordMetric("sampleLoss", step, lossEma);
}

function setMetricText(element, value, format) {
	if (Number.isFinite(value)) element.textContent = format(value);
}

function consumeValidation({ step, metrics }) {
	globalThis.__dynaworldValidationMetrics = { step, ...metrics };
	setMetricText(values.gridLoss, metrics.gridLoss, (value) => value.toFixed(6));
	setMetricText(values.trainMae, metrics.gridMae, (value) => value.toFixed(4));
	setMetricText(values.trainPsnr, metrics.gridPsnr, (value) => `${value.toFixed(1)} dB`);
	setMetricText(values.trainSsim, metrics.gridSsim, (value) => value.toFixed(3));
	setMetricText(values.heldoutLoss, metrics.heldoutLoss, (value) => value.toFixed(6));
	setMetricText(values.heldoutMae, metrics.heldoutMae, (value) => value.toFixed(4));
	setMetricText(values.heldoutPsnr, metrics.heldoutPsnr, (value) => `${value.toFixed(1)} dB`);
	setMetricText(values.heldoutSsim, metrics.heldoutSsim, (value) => value.toFixed(3));
	setMetricText(values.heldoutCoverage, metrics.heldoutCoverage, (value) => `${(value * 100).toFixed(1)}%`);
	if (Number.isFinite(metrics.gridDetailMae) && Number.isFinite(metrics.heldoutDetailMae)) {
		values.detailMae.textContent = `${metrics.gridDetailMae.toFixed(3)} / `
			+ `${metrics.heldoutDetailMae.toFixed(3)}`;
	}
	if (Number.isFinite(metrics.gridLowPassPsnr) && Number.isFinite(metrics.heldoutLowPassPsnr)) {
		values.lowPassPsnr.textContent = `${metrics.gridLowPassPsnr.toFixed(1)} / `
			+ `${metrics.heldoutLowPassPsnr.toFixed(1)} dB`;
	}
	if (Number.isFinite(metrics.weakestTrainCameraPsnr)
		&& Number.isFinite(metrics.strongestTrainCameraPsnr)) {
		values.cameraPsnr.textContent = `${metrics.weakestTrainCamera ?? "worst"} `
			+ `${metrics.weakestTrainCameraPsnr.toFixed(1)}–${metrics.strongestTrainCameraPsnr.toFixed(1)}`;
	}
	setMetricText(values.motionLoss, metrics.motionLoss, (value) => value.toFixed(6));
	setMetricText(values.motionCoverage, metrics.motionCoverage, (value) => `${(value * 100).toFixed(1)}%`);
	setMetricText(values.staticCoverage, metrics.staticCoverage, (value) => `${(value * 100).toFixed(1)}%`);
	setMetricText(values.peakAlpha, metrics.motionMaxAlpha, (value) => `${(value * 100).toFixed(1)}%`);
	setMetricText(values.active, metrics.activeSplats,
		(value) => `${value}/${trainerCapacity || controls.splats.value}`);
	setMetricText(values.dynamicSplats, metrics.dynamicSplats,
		(value) => `${value}/${metrics.temporalAnalyzedSplats ?? metrics.activeSplats}`);
	setMetricText(values.persistentSplats, metrics.persistentSplats,
		(value) => `${value}/${metrics.temporalAnalyzedSplats ?? metrics.activeSplats}`);
	setMetricText(values.staticMixP50, metrics.staticMixP50, (value) => value.toFixed(3));
	setMetricText(values.edgeSupport, metrics.meanEdgeTemporalSupport,
		(value) => `${(value * 100).toFixed(1)}%`);
	setMetricText(values.aspectP90, metrics.aspectP90,
		(value) => `${value.toFixed(2)}:1 · ${((metrics.aspectCapFraction ?? 0) * 100).toFixed(0)}%`);
	setMetricText(values.rasterDead, metrics.rasterDeadSplats,
		(value) => `${value}/${trainerCapacity || controls.splats.value}`);
	setMetricText(values.meanOpacity, metrics.meanOpacity, (value) => `${(value * 100).toFixed(1)}%`);
	setMetricText(values.meanRadius, metrics.meanRadius, (value) => value.toFixed(4));
	setMetricText(values.meanAspect, metrics.meanAspectRatio, (value) => `${value.toFixed(2)}:1`);
	setMetricText(values.recycled, metrics.totalRecycled, (value) => String(value));
	setMetricText(values.parameterDelta, metrics.parameterDelta, (value) => value.toExponential(2));
	const updates = metrics.parameterUpdateRatios ?? {};
	setMetricText(values.centerUpdate, updates.center?.updateRms, (value) => value.toExponential(1));
	setMetricText(values.motionUpdate, Math.max(
		updates.staticMix?.updateRms ?? Number.NaN,
		updates.velocity?.updateRms ?? Number.NaN,
		updates.timeCenter?.updateRms ?? Number.NaN,
		updates.harmonic?.updateRms ?? Number.NaN,
	), (value) => value.toExponential(1));
	setMetricText(values.scaleUpdate, updates.logScale?.updateRms, (value) => value.toExponential(1));
	setMetricText(values.rotationUpdate, updates.rotation?.updateRms, (value) => value.toExponential(1));
	setMetricText(values.colorUpdate, updates.color?.updateRms, (value) => value.toExponential(1));
	setMetricText(values.opacityUpdate, updates.opacity?.updateRms, (value) => value.toExponential(1));
	if (metrics.validationContract) {
		const trainViews = metrics.validationContract.trainViews?.length ?? 0;
		const heldoutViews = metrics.validationContract.heldoutViews?.length ?? 0;
		const seconds = Number(metrics.validationDurationMs) / 1000;
		values.validation.textContent = `full ${trainViews}+${heldoutViews} · ${seconds.toFixed(1)}s`;
	}
	recordMetric("trainLoss", step, metrics.gridLoss);
	recordMetric("heldoutLoss", step, metrics.heldoutLoss);
	recordMetric("trainPsnr", step, metrics.gridPsnr);
	recordMetric("heldoutPsnr", step, metrics.heldoutPsnr);
	recordMetric("trainSsim", step, metrics.gridSsim);
	recordMetric("heldoutSsim", step, metrics.heldoutSsim);
}

function selectedViewDataset() {
	const index = Number(controls.renderCamera.value || 0);
	return dataset?.viewDatasets?.[index] ?? dataset?.previewViews?.[index] ?? dataset;
}

function updateTargetCanvas() {
	if (!dataset) return;
	let targetView = controls.targetView.value;
	if (workerClient && targetView === "validation_error") {
		targetView = "rgb";
		controls.targetView.value = "rgb";
		setStatus("Validation error images are not in the nonblocking worker snapshot protocol; scalar validation remains live.");
	}
	const time = Number(controls.time.value);
	const selected = selectedViewDataset();
	const frame = drawTargetFrame(targetCanvas, selected, time, { view: targetView });
	const camera = dataset.cameras?.[Number(controls.renderCamera.value || 0)];
	$("targetFrameValue").textContent = `${camera ? `${camera.name} ${camera.role}` : "target"} f${frame}`;
	const previews = dataset.previewViews ?? [];
	$("angleStrip").hidden = !previews.length;
	for (let index = 0; index < comparisonCanvases.length; index += 1) {
		const preview = previews[index];
		comparisonCanvases[index].hidden = !preview;
		comparisonFrameLabels[index].textContent = preview
			? `${preview.label} f${drawTargetFrame(comparisonCanvases[index], preview, time)}` : "--";
	}
}

function configureCameraUi() {
	const cameras = dataset.cameras ?? [{ name: "Source", role: "train" }];
	controls.renderCamera.replaceChildren(...cameras.map((camera, index) => {
		const option = document.createElement("option"); option.value = String(index);
		option.textContent = `${camera.name} (${camera.role})`; return option;
	}));
	controls.renderCamera.value = String(dataset.comparisonViewIndices?.[0] ?? 0);
	$("resultAngleLabels").hidden = (dataset.viewCount ?? 1) < 2;
	const renderedIndices = cameras.length > 1 ? dataset.comparisonViewIndices : [0];
	for (let panel = 0; panel < 3; panel += 1) {
		const camera = cameras[renderedIndices[panel]];
		resultViewLabels[panel].textContent = camera?.name ?? "--";
		resultViewRoles[panel].textContent = camera?.role === "heldout" ? "Heldout" : `Train ${String.fromCharCode(65 + panel)}`;
	}
	const notice = $("renderContractNotice");
	notice.hidden = true;
}

function syncWorkerOptions(force = false) {
	if (!workerClient) return;
	const nextTrain = trainOptions(); const trainKey = JSON.stringify(nextTrain);
	if (force || trainKey !== lastTrainOptionsKey) {
		workerClient.setTrainOptions(nextTrain); lastTrainOptionsKey = trainKey;
	}
	const nextRender = renderOptions(); const renderKey = JSON.stringify(nextRender);
	if (force || renderKey !== lastRenderOptionsKey) {
		workerClient.setRenderOptions(nextRender); lastRenderOptionsKey = renderKey;
	}
}

function resizeWorkerCanvas() {
	if (!workerClient) return;
	const rect = renderCanvas.getBoundingClientRect();
	const scale = Math.min(window.devicePixelRatio || 1, 1.25, MAX_RENDER_WIDTH / Math.max(1, rect.width));
	workerClient.resize(Math.max(1, Math.floor(rect.width * scale)), Math.max(1, Math.floor(rect.height * scale)));
}

function bindWorkerEvents(client) {
	client.addEventListener(WorkerEvent.METRICS, ({ detail }) => consumeSampleMetric(detail));
	client.addEventListener(WorkerEvent.VALIDATION, ({ detail }) => consumeValidation(detail));
	client.addEventListener(WorkerEvent.ERROR, ({ detail }) => {
		setRunning(false);
		setStatus([detail.message || "Training worker failed.", detail.stack].filter(Boolean).join("\n"));
		console.error(detail.error ?? detail);
	});
	client.addEventListener(WorkerEvent.CAPABILITY, ({ detail }) => {
		setStatus(`${detail.capability}: ${detail.available ? "available" : detail.reason}`);
	});
}

async function initWorkerTrainer() {
	workerClient = createNonblockingTrainer();
	bindWorkerEvents(workerClient);
	const ready = await workerClient.init({
		dataset, canvas: renderCanvas,
		trainerOptions: { backend: controls.backend.value, splatCount: Number(controls.splats.value),
			growthCapacity: sampledBackendSelected() ? null : Number(controls.growthCapacity.value),
			checkpointPrecision: controls.precision.value,
			staticWarmupSteps: controls.staticWarmup.checked && !sampledBackendSelected()
				? STATIC_WARMUP_STEPS : 0 },
		trainOptions: trainOptions(), renderOptions: renderOptions(),
		schedule: { validationEvery: 0, renderFps: RENDER_FPS },
	});
	trainerCapacity = ready.backend?.capacity ?? Number(controls.splats.value);
	trainerStaticWarmupSteps = ready.backend?.memoryPlan?.staticWarmupSteps ?? 0;
	document.documentElement.dataset.trainerBackend = ready.backend?.id ?? "unknown";
	values.gpu.textContent = ready.adapter ?? "WebGPU";
	values.runtime.textContent = ready.capabilities.offscreenRender
		? `${ready.backend?.label ?? "worker"} + render` : `${ready.backend?.label ?? "worker"} optimizer`;
	values.representation.textContent = "Trajectory 3DGS";
	values.representation.title = ready.backend?.representation ?? "trajectory-gated dynamic 3DGS";
	$("motionCoverageLabel").textContent = ready.backend?.sampledControls ? "Motion Cov" : "Train Cov";
	values.shared.textContent = ready.capabilities.sharedStatus ? "atomic SAB" : "messages";
	values.validation.textContent = ready.capabilities.validationWorker ? "separate worker" : "unavailable";
	if (!ready.capabilities.offscreenRender) {
		$("renderContractNotice").hidden = false;
		$("renderContractNotice").textContent = "OffscreenCanvas is unavailable; optimization remains worker-owned but live rendering is disabled.";
	}
	resizeWorkerCanvas(); syncWorkerOptions(true);
	workerClient.requestValidation();
	lastValidationRequestStep = 0;
	const cameraBatch = ready.cameraBatch;
	const heldoutCamera = dataset.cameras?.[dataset.heldoutViewIndex];
	const heldoutDescription = heldoutCamera
		? `${heldoutCamera.name} is held out`
		: "no heldout camera is configured";
	values.splatCount.textContent = trainerCapacity === Number(controls.splats.value)
		? String(trainerCapacity) : `${controls.splats.value} → ${trainerCapacity}`;
	const checkpointPrecision = ready.backend?.memoryPlan?.checkpointPrecision;
	const allocatedBytes = ready.backend?.memoryPlan?.allocatedBytes;
	if (Number.isFinite(allocatedBytes)) {
		values.gpuMemory.textContent = `${(allocatedBytes / (1024 * 1024)).toFixed(1)} MiB`;
		values.gpuMemory.title = Object.entries(ready.backend.memoryPlan.bufferBytes ?? {})
			.map(([name, bytes]) => `${name}: ${(bytes / (1024 * 1024)).toFixed(2)} MiB`)
			.join("\n");
	}
	const objective = controls.motionWeighting.checked && !sampledBackendSelected()
		? `motion-weighted ${ready.backend?.objective ?? "training"}`
		: ready.backend?.objective ?? "training";
	const background = controls.randomBackground.checked && !sampledBackendSelected()
		? "random-RGB train underlay · "
		: "";
	setStatus(`Ready: ${ready.backend?.label ?? "WebGPU"} · `
		+ `${ready.backend?.representation ?? "trajectory-gated dynamic 3DGS"} · `
		+ `${objective} · `
		+ `${checkpointPrecision ? `${checkpointPrecision} checkpoints · ` : ""}`
		+ background
		+ `${trainerStaticWarmupSteps ? `${trainerStaticWarmupSteps}-step train-only static warmup · ` : ""}`
		+ `${cameraBatch?.camerasPerStep ?? 1} of ${cameraBatch?.trainViewCount ?? 17} train cameras per step; `
		+ `${heldoutDescription}; init ${dataset.seedProvenance?.train_only_verified
			? "train-only verified" : "external/unverified"}.`);
}

async function initLocalTrainer() {
	localTrainer = new DynamicSplatWebGpuTrainer(renderCanvas);
	await localTrainer.init(dataset, { splatCount: Number(controls.splats.value) });
	values.gpu.textContent = localTrainer.adapterName;
	values.runtime.textContent = "main-thread fallback";
	values.shared.textContent = "n/a";
	values.validation.textContent = "local";
	localTrainer.render(Number(controls.time.value), currentModelMode(), currentTemporalSigma(), currentRenderMode());
	setStatus("Single-view compatibility mode. Multicamera datasets use the nonblocking worker runtime.");
}

function saveUiState() {
	const state = {};
	for (const [name, control] of Object.entries(controls)) {
		if (!(control instanceof HTMLInputElement || control instanceof HTMLSelectElement)) continue;
		state[name] = control.type === "checkbox" ? control.checked : control.value;
	}
	sessionStorage.setItem(UI_STATE_KEY, JSON.stringify(state));
}

function restoreUiState() {
	try {
		const state = JSON.parse(sessionStorage.getItem(UI_STATE_KEY) || "null");
		sessionStorage.removeItem(UI_STATE_KEY);
		if (!state) return;
		for (const [name, saved] of Object.entries(state)) {
			const control = controls[name]; if (!control) continue;
			if (control.type === "checkbox") control.checked = Boolean(saved); else control.value = saved;
		}
	} catch (error) {
		console.warn("Could not restore trainer controls.", error);
	}
}

async function resetTrainer() {
	if (workerClient) {
		saveUiState(); location.reload();
		return;
	}
	if (!dataset || booting) return;
	booting = true; setRunning(false);
	for (const control of [controls.run, controls.step, controls.reset]) control.disabled = true;
	try {
		localTrainer?.dispose(); localTrainer = null; lossEma = null; resetMetrics();
		await initLocalTrainer();
	} catch (error) {
		setStatus(error instanceof Error ? error.message : String(error)); console.error(error);
	} finally {
		booting = false;
		for (const control of [controls.run, controls.step, controls.reset]) control.disabled = false;
	}
}

async function readLocalLoss() {
	if (!localTrainer || localLossPending || localTrainer.stepCount - lastLocalLossStep < 16) return;
	localLossPending = true; lastLocalLossStep = localTrainer.stepCount;
	try { consumeSampleMetric({ step: localTrainer.stepCount, loss: await localTrainer.readLoss(trainOptions()) }); }
	finally { localLossPending = false; }
}

function tickLocalTrainer() {
	if (!localTrainer) return;
	if (running) for (let iteration = 0; iteration < 2; iteration += 1) localTrainer.trainStep(trainOptions());
	if (controls.live.checked) localTrainer.render(Number(controls.time.value), currentModelMode(),
		currentTemporalSigma(), currentRenderMode());
	void readLocalLoss();
}

function updateFrameDiagnostics(now, delta) {
	frameDurations.push(delta); if (frameDurations.length > 180) frameDurations.shift();
	const sorted = [...frameDurations].sort((a, b) => a - b);
	const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? delta;
	const uiFps = 1000 / Math.max(1, frameDurations.reduce((sum, value) => sum + value, 0) / frameDurations.length);
	values.fps.textContent = workerClient && controls.live.checked
		? `${RENDER_FPS} GPU · ${Math.round(uiFps)} UI` : controls.live.checked ? `${Math.round(uiFps)} fps` : "off";
	globalThis.__dynaworldDiagnostics = { at: now, uiFps, uiP95Ms: p95,
		completedStepsPerSecond: workerClient?.getStatus()?.stepsPerSecond ?? 0, step: currentStep(),
		metricCounts: Object.fromEntries(Object.entries(metricHistory)
			.map(([name, points]) => [name, points.length])) };
}

function frameLoop(now) {
	const delta = Math.max(1, now - lastFrameAt); lastFrameAt = now;
	if (controls.loop.checked && dataset) {
		controls.time.value = ((Number(controls.time.value) + delta / 1000 * Number(controls.speed.value)) % 1).toFixed(3);
	}
	if (workerClient) {
		const status = workerClient.getStatus();
		if (status) {
			values.step.textContent = String(status.step);
			values.phase.textContent = status.step < trainerStaticWarmupSteps ? "static init" : "dynamic fit";
			values.stepRate.textContent = Number.isFinite(status.stepsPerSecond) ? status.stepsPerSecond.toFixed(1) : "--";
			if (status.lastMetricStep > lastStatusMetricStep && Number.isFinite(status.loss)) {
				lastStatusMetricStep = status.lastMetricStep;
			}
		}
		const validationStep = status?.step ?? 0;
		if (controls.fullMetrics.checked
			&& validationStep - lastValidationRequestStep >= VALIDATION_STEP_INTERVAL
			&& !(status?.flags & StatusFlag.VALIDATION_PENDING)) {
			workerClient.requestValidation();
			lastValidationRequestStep = validationStep;
		}
		syncWorkerOptions();
	} else {
		tickLocalTrainer(); values.step.textContent = String(currentStep());
	}
	updateControlLabels();
	if (controls.live.checked && now - lastTargetDrawAt >= 1000 / 30) {
		lastTargetDrawAt = now; updateTargetCanvas();
	}
	updateFrameDiagnostics(now, delta);
	animationHandle = requestAnimationFrame(frameLoop);
}

async function boot() {
	restoreUiState(); updateControlLabels(); resetMetrics();
	for (const control of [controls.run, controls.step, controls.reset]) control.disabled = true;
	try {
		dataset = await loadPresetDataset();
		const trainCount = dataset.trainViewCount ?? 1;
		const poseSource = dataset.datasetContract?.pose_source;
		const calibrationLabel = poseSource?.endsWith("_v2") ? "LLFF/OpenCV v2" : poseSource;
		$("datasetName").textContent = dataset.datasetContract
			? `${dataset.name} · ${trainCount} train / ${dataset.viewCount - trainCount} heldout`
				+ `${calibrationLabel ? ` · ${calibrationLabel}` : ""}`
			: dataset.name;
		$("datasetName").title = poseSource ?? "No calibrated pose source declared";
		values.splatCount.textContent = controls.splats.value;
		values.motionSamples.textContent = String(dataset.motionSamples?.length ?? 0);
		values.staticSamples.textContent = String(dataset.staticSamples?.length ?? 0);
		values.seedProvenance.textContent = dataset.seedProvenance?.train_only_verified
			? "train-only verified" : "unverified";
		configureCameraUi(); updateTargetCanvas();
		if ((dataset.viewCount ?? 1) > 1) await initWorkerTrainer(); else await initLocalTrainer();
	} catch (error) {
		setStatus(error instanceof Error ? error.message : String(error)); console.error(error);
	} finally {
		booting = false;
		for (const control of [controls.run, controls.step, controls.reset]) control.disabled = false;
	}
	animationHandle = requestAnimationFrame((now) => { lastFrameAt = now; frameLoop(now); });
}

controls.run.addEventListener("click", () => setRunning(!running));
controls.step.addEventListener("click", () => {
	if (workerClient) { workerClient.step(1); workerClient.requestValidation(); }
	else if (localTrainer) { localTrainer.trainStep(trainOptions()); void readLocalLoss(); }
});
controls.reset.addEventListener("click", () => { void resetTrainer(); });
controls.splats.addEventListener("change", () => { void resetTrainer(); });
controls.growthCapacity.addEventListener("change", () => { void resetTrainer(); });
controls.backend.addEventListener("change", () => { updateControlLabels(); void resetTrainer(); });
controls.precision.addEventListener("change", () => { void resetTrainer(); });
controls.staticWarmup.addEventListener("change", () => { void resetTrainer(); });
controls.mode.addEventListener("change", () => { syncWorkerOptions(true); updateControlLabels(); });
for (const control of [controls.time, controls.speed, controls.temporal, controls.lr, controls.samples,
	controls.motionMix, controls.staticMix, controls.supportGuard]) {
	control.addEventListener("input", () => { updateControlLabels(); syncWorkerOptions(); updateTargetCanvas(); });
}
for (const control of [controls.loop, controls.live, controls.temporalSchedule, controls.lrSchedule,
	controls.motionWeighting, controls.randomBackground]) {
	control.addEventListener("change", () => { updateControlLabels(); syncWorkerOptions(true); });
}
controls.fullMetrics.addEventListener("change", () => {
	if (controls.fullMetrics.checked) {
		lastValidationRequestStep = currentStep();
		workerClient?.requestValidation();
	}
});
controls.targetView.addEventListener("change", updateTargetCanvas);
controls.renderCamera.addEventListener("change", () => { updateTargetCanvas(); syncWorkerOptions(true); });
controls.resultView.addEventListener("change", () => syncWorkerOptions(true));
chartElements.lossScale.addEventListener("click", () => {
	lossLogScale = !lossLogScale;
	chartElements.lossScale.setAttribute("aria-pressed", String(lossLogScale));
	drawMetricCharts();
});

new ResizeObserver(() => resizeWorkerCanvas()).observe(renderCanvas);
const chartObserver = new ResizeObserver(drawMetricCharts);
for (const canvas of [chartElements.loss, chartElements.psnr, chartElements.ssim]) chartObserver.observe(canvas);
window.addEventListener("beforeunload", () => {
	if (animationHandle) cancelAnimationFrame(animationHandle);
	workerClient?.dispose(); localTrainer?.dispose();
});

void boot();
