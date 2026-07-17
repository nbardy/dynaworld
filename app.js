import { drawTargetFrame, loadPresetDataset } from "./dataset.js?v=20260710-converge59";
import { DynamicSplatWebGpuTrainer } from "./trainerWebGpu.js?v=20260710-converge59";

const renderCanvas = document.getElementById("renderCanvas");
const targetCanvas = document.getElementById("targetCanvas");
const sourceViewCanvas = document.getElementById("sourceViewCanvas");
const targetViewCanvas = document.getElementById("targetViewCanvas");
const angleStrip = document.getElementById("angleStrip");
const runButton = document.getElementById("runButton");
const stepButton = document.getElementById("stepButton");
const resetButton = document.getElementById("resetButton");
const modeSelect = document.getElementById("modeSelect");
const splatSlider = document.getElementById("splatSlider");
const timeSlider = document.getElementById("timeSlider");
const timeLoopToggle = document.getElementById("timeLoopToggle");
const timeSpeedSlider = document.getElementById("timeSpeedSlider");
const targetViewSelect = document.getElementById("targetViewSelect");
const resultViewSelect = document.getElementById("resultViewSelect");
const temporalSlider = document.getElementById("temporalSlider");
const lrSlider = document.getElementById("lrSlider");
const samplesSlider = document.getElementById("samplesSlider");
const motionMixSlider = document.getElementById("motionMixSlider");
const staticMixSlider = document.getElementById("staticMixSlider");
const supportGuardSlider = document.getElementById("supportGuardSlider");
const splatSliderValue = document.getElementById("splatSliderValue");
const timeValue = document.getElementById("timeValue");
const timeSpeedValue = document.getElementById("timeSpeedValue");
const temporalValue = document.getElementById("temporalValue");
const lrValue = document.getElementById("lrValue");
const samplesValue = document.getElementById("samplesValue");
const motionMixValue = document.getElementById("motionMixValue");
const staticMixValue = document.getElementById("staticMixValue");
const supportGuardValue = document.getElementById("supportGuardValue");
const stepValue = document.getElementById("stepValue");
const stepRateValue = document.getElementById("stepRateValue");
const lossValue = document.getElementById("lossValue");
const gridLossValue = document.getElementById("gridLossValue");
const valMaeValue = document.getElementById("valMaeValue");
const valPsnrValue = document.getElementById("valPsnrValue");
const valSsimValue = document.getElementById("valSsimValue");
const motionLossValue = document.getElementById("motionLossValue");
const motionCoverageValue = document.getElementById("motionCoverageValue");
const staticCoverageValue = document.getElementById("staticCoverageValue");
const motionMaxAlphaValue = document.getElementById("motionMaxAlphaValue");
const activeSplatValue = document.getElementById("activeSplatValue");
const meanOpacityValue = document.getElementById("meanOpacityValue");
const meanRadiusValue = document.getElementById("meanRadiusValue");
const splatValue = document.getElementById("splatValue");
const recycledSplatValue = document.getElementById("recycledSplatValue");
const parameterDeltaValue = document.getElementById("parameterDeltaValue");
const motionSampleValue = document.getElementById("motionSampleValue");
const staticSampleValue = document.getElementById("staticSampleValue");
const gpuValue = document.getElementById("gpuValue");
const fpsValue = document.getElementById("fpsValue");
const statusText = document.getElementById("statusText");
const datasetName = document.getElementById("datasetName");
const targetFrameValue = document.getElementById("targetFrameValue");
const sourceViewFrameValue = document.getElementById("sourceViewFrameValue");
const targetViewFrameValue = document.getElementById("targetViewFrameValue");

let dataset = null;
let trainer = null;
let running = false;
let lastFrameMs = performance.now();
let lastLossStep = -1;
let lastGridLossStep = -1;
let lastRateStep = 0;
let lastRateMs = performance.now();
let lossEma = null;
let animationHandle = 0;
let booting = false;
let gridLossBusy = false;
let validationEpoch = 0;
let errorMapBusy = false;
let lastErrorMapMs = 0;
let lastErrorMapKey = "";

function setStatus(message) {
	statusText.textContent = message;
}

function previewTime() {
	return Number(timeSlider.value);
}

function currentLoopSpeed() {
	return Number(timeSpeedSlider.value);
}

function currentModelMode() {
	return modeSelect.value === "dynamic_splats" ? 1 : 0;
}

function currentSplatCount() {
	return Number(splatSlider.value);
}

function currentTemporalSigma() {
	return Number(temporalSlider.value);
}

function currentMotionMix() {
	return Number(motionMixSlider.value);
}

function currentStaticMix() {
	return Number(staticMixSlider.value);
}

function currentSupportGuard() {
	return Number(supportGuardSlider.value);
}

function effectiveMotionMix() {
	return Math.min(currentMotionMix(), 1 - currentStaticMix());
}

function currentRenderMode() {
	if (resultViewSelect?.value === "alpha_support") {
		return 2;
	}
	return resultViewSelect?.value === "dynamic_residual" ? 1 : 0;
}

function updateSliderLabels() {
	splatSliderValue.textContent = splatSlider.value;
	timeValue.textContent = previewTime().toFixed(3);
	timeSpeedValue.textContent = `${currentLoopSpeed().toFixed(2)}x`;
	temporalValue.textContent = currentTemporalSigma().toFixed(2);
	lrValue.textContent = `${Number(lrSlider.value).toFixed(2)}x`;
	samplesValue.textContent = samplesSlider.value;
	motionMixValue.textContent = `${Math.round(effectiveMotionMix() * 100)}%`;
	staticMixValue.textContent = `${Math.round(currentStaticMix() * 100)}%`;
	supportGuardValue.textContent = `${Math.round(currentSupportGuard() * 100)}%`;
}

function setRunning(value) {
	running = value;
	runButton.textContent = running ? "Pause" : "Start";
	runButton.dataset.running = running ? "true" : "false";
}

function drawValidationErrorCanvas(result) {
	const ctx = targetCanvas.getContext("2d");
	if (!ctx) {
		return;
	}
	targetCanvas.width = result.width;
	targetCanvas.height = result.height;
	const image = new ImageData(result.data, result.width, result.height);
	ctx.putImageData(image, 0, 0);
	targetFrameValue.textContent = `frame ${result.frame} error ${result.meanAbs.toFixed(4)}`;
}

async function refreshValidationErrorCanvas(time) {
	if (!trainer || errorMapBusy) {
		return;
	}
	const frame = dataset.frameCount <= 1 ? 0 : Math.round(time * (dataset.frameCount - 1));
	const key = `${trainer.stepCount}:${frame}:${currentModelMode()}:${currentTemporalSigma().toFixed(3)}`;
	const now = performance.now();
	if (key === lastErrorMapKey || now - lastErrorMapMs < 650) {
		return;
	}
	errorMapBusy = true;
	const activeTrainer = trainer;
	const activeEpoch = validationEpoch;
	try {
		const result = await activeTrainer.readPreviewErrorImage({
			time,
			modelMode: currentModelMode(),
			temporalSigma: currentTemporalSigma(),
		});
		if (trainer !== activeTrainer || validationEpoch !== activeEpoch) {
			return;
		}
		drawValidationErrorCanvas(result);
		lastErrorMapKey = key;
		lastErrorMapMs = performance.now();
	} catch (error) {
		if (trainer === activeTrainer && validationEpoch === activeEpoch) {
			console.warn("Validation error image failed.", error);
		}
	} finally {
		if (validationEpoch === activeEpoch) {
			errorMapBusy = false;
		}
	}
}

function updateTargetCanvas() {
	if (!dataset) {
		return;
	}
	const targetView = targetViewSelect?.value ?? "rgb";
	const time = previewTime();
	if (targetView === "validation_error") {
		if (errorMapBusy) {
			targetFrameValue.textContent = "updating error";
		} else if (!lastErrorMapKey) {
			targetFrameValue.textContent = "validation error";
		}
		void refreshValidationErrorCanvas(time);
	} else {
		const frame = drawTargetFrame(targetCanvas, dataset, time, { view: targetView });
		targetFrameValue.textContent = targetView === "motion_residual" ? `frame ${frame} residual` : `frame ${frame}`;
	}
	const views = dataset.previewViews ?? [];
	angleStrip.hidden = views.length === 0;
	const previewPairs = [
		[sourceViewCanvas, sourceViewFrameValue, views[0]],
		[targetViewCanvas, targetViewFrameValue, views[1]],
	];
	for (const [canvas, label, view] of previewPairs) {
		if (!canvas || !label) {
			continue;
		}
		if (!view) {
			canvas.hidden = true;
			label.textContent = "--";
			continue;
		}
		canvas.hidden = false;
		const viewFrame = drawTargetFrame(canvas, view, time, { view: "rgb" });
		label.textContent = `${view.label ?? "View"} f${viewFrame}`;
	}
}

function advancePreviewTime(deltaMs) {
	if (!timeLoopToggle.checked || !dataset) {
		return false;
	}
	const next = (previewTime() + (deltaMs / 1000) * currentLoopSpeed()) % 1;
	timeSlider.value = next.toFixed(3);
	updateSliderLabels();
	updateTargetCanvas();
	return true;
}

async function readLossIfReady(force = false) {
	if (!trainer) {
		return;
	}
	if (!force && trainer.stepCount === lastLossStep) {
		return;
	}
	if (!force && trainer.stepCount - lastLossStep < 16) {
		return;
	}
	lastLossStep = trainer.stepCount;
	try {
		const loss = await trainer.readLoss();
		if (Number.isFinite(loss)) {
			lossEma = lossEma == null ? loss : lossEma * 0.82 + loss * 0.18;
			lossValue.textContent = lossEma.toFixed(5);
		}
	} catch (error) {
		console.warn("Loss readback failed.", error);
	}
}

async function readGridLossIfReady(force = false) {
	if (!trainer) {
		return;
	}
	if (gridLossBusy) {
		if (!force) {
			return;
		}
		while (gridLossBusy && trainer) {
			await new Promise((resolve) => setTimeout(resolve, 16));
		}
		if (!trainer) {
			return;
		}
	}
	if (!force && trainer.stepCount - lastGridLossStep < 128) {
		return;
	}
	const activeTrainer = trainer;
	const activeEpoch = validationEpoch;
	const step = activeTrainer.stepCount;
	lastGridLossStep = step;
	gridLossBusy = true;
	try {
		const metrics = await activeTrainer.readValidationMetrics({
			modelMode: currentModelMode(),
			temporalSigma: currentTemporalSigma(),
			gridSize: 32,
		});
		if (trainer !== activeTrainer || validationEpoch !== activeEpoch) {
			return;
		}
		if (Number.isFinite(metrics.gridLoss)) {
			gridLossValue.textContent = metrics.gridLoss.toFixed(6);
		}
		if (Number.isFinite(metrics.gridMae)) {
			valMaeValue.textContent = metrics.gridMae.toFixed(4);
		}
		if (Number.isFinite(metrics.gridPsnr)) {
			valPsnrValue.textContent = `${metrics.gridPsnr.toFixed(1)} dB`;
		}
		if (Number.isFinite(metrics.gridSsim)) {
			valSsimValue.textContent = metrics.gridSsim.toFixed(3);
		}
		if (Number.isFinite(metrics.motionLoss)) {
			motionLossValue.textContent = metrics.motionLoss.toFixed(6);
		}
		if (Number.isFinite(metrics.motionCoverage)) {
			motionCoverageValue.textContent = `${(metrics.motionCoverage * 100).toFixed(1)}%`;
		}
		if (Number.isFinite(metrics.staticCoverage)) {
			staticCoverageValue.textContent = `${(metrics.staticCoverage * 100).toFixed(1)}%`;
		}
		if (Number.isFinite(metrics.motionMaxAlpha)) {
			motionMaxAlphaValue.textContent = `${(metrics.motionMaxAlpha * 100).toFixed(1)}%`;
		}
		if (Number.isFinite(metrics.activeSplats)) {
			activeSplatValue.textContent = `${metrics.activeSplats}/${trainer.splatCount}`;
		}
		if (Number.isFinite(metrics.meanOpacity)) {
			meanOpacityValue.textContent = `${(metrics.meanOpacity * 100).toFixed(1)}%`;
		}
		if (Number.isFinite(metrics.meanRadius)) {
			meanRadiusValue.textContent = metrics.meanRadius.toFixed(4);
		}
		if (Number.isFinite(metrics.parameterDelta)) {
			parameterDeltaValue.textContent = metrics.parameterDelta.toExponential(2);
		}
	} catch (error) {
		if (trainer === activeTrainer && validationEpoch === activeEpoch) {
			console.warn("Grid loss readback failed.", error);
		}
	} finally {
		if (validationEpoch === activeEpoch) {
			gridLossBusy = false;
		}
	}
}

function renderOnce() {
	if (!trainer?.dataset || !trainer.device) {
		return;
	}
	trainer.render(previewTime(), currentModelMode(), currentTemporalSigma(), currentRenderMode());
	stepValue.textContent = String(trainer.stepCount);
}

async function resetTrainer() {
	if (booting) {
		return;
	}
	booting = true;
	setRunning(false);
	runButton.disabled = true;
	stepButton.disabled = true;
	resetButton.disabled = true;
	setStatus("Preparing WebGPU trainer.");
	let nextTrainer = null;
	try {
		validationEpoch += 1;
		gridLossBusy = false;
		errorMapBusy = false;
		lastErrorMapKey = "";
		const oldTrainer = trainer;
		trainer = null;
		oldTrainer?.dispose();
		nextTrainer = new DynamicSplatWebGpuTrainer(renderCanvas);
		await nextTrainer.init(dataset, { splatCount: currentSplatCount() });
		trainer = nextTrainer;
		nextTrainer = null;
		splatValue.textContent = String(trainer.splatCount);
		motionSampleValue.textContent = String(dataset.motionSamples?.length ?? 0);
		staticSampleValue.textContent = String(dataset.staticSamples?.length ?? 0);
		gpuValue.textContent = trainer.adapterName;
		lastLossStep = -1;
		lastGridLossStep = -1;
		lastRateStep = 0;
		lastRateMs = performance.now();
		lossEma = null;
		lossValue.textContent = "--";
		stepRateValue.textContent = "--";
		gridLossValue.textContent = "--";
		valMaeValue.textContent = "--";
		valPsnrValue.textContent = "--";
		valSsimValue.textContent = "--";
		motionLossValue.textContent = "--";
		motionCoverageValue.textContent = "--";
		staticCoverageValue.textContent = "--";
		motionMaxAlphaValue.textContent = "--";
		activeSplatValue.textContent = "--";
		meanOpacityValue.textContent = "--";
		meanRadiusValue.textContent = "--";
		recycledSplatValue.textContent = "0";
		parameterDeltaValue.textContent = "0";
		stepValue.textContent = "0";
		renderOnce();
		void readGridLossIfReady(true);
		setStatus("Ready.");
	} catch (error) {
		nextTrainer?.dispose();
		trainer?.dispose();
		trainer = null;
		setStatus(error instanceof Error ? error.message : String(error));
		console.error(error);
	} finally {
		runButton.disabled = false;
		stepButton.disabled = false;
		resetButton.disabled = false;
		booting = false;
	}
}

async function boot() {
	updateSliderLabels();
	runButton.disabled = true;
	stepButton.disabled = true;
	resetButton.disabled = true;
	try {
		dataset = await loadPresetDataset();
		datasetName.textContent = dataset.name;
		setStatus(`Loaded ${dataset.width}x${dataset.height}x${dataset.frameCount} target.`);
		updateTargetCanvas();
		await resetTrainer();
	} catch (error) {
		setStatus(error instanceof Error ? error.message : String(error));
		console.error(error);
	} finally {
		runButton.disabled = false;
		stepButton.disabled = false;
		resetButton.disabled = false;
	}
}

async function stepTrainer(iterations = 1) {
	if (!trainer?.dataset || !trainer.device) {
		return;
	}
	const learningRate = Number(lrSlider.value);
	const samplesPerStep = Number(samplesSlider.value);
	const modelMode = currentModelMode();
	const temporalSigma = currentTemporalSigma();
	const motionSampleRate = currentMotionMix();
	const staticSampleRate = currentStaticMix();
	const motionCoverageTarget = currentSupportGuard();
	for (let i = 0; i < iterations; i += 1) {
		trainer.trainStep({
			learningRate,
			samplesPerStep,
			modelMode,
			temporalSigma,
			motionSampleRate,
			staticSampleRate,
			motionCoverageTarget,
		});
	}
	const recycled = await trainer.maintainDensity({ modelMode, temporalSigma });
	if (recycled > 0) {
		recycledSplatValue.textContent = String(trainer.totalRecycled);
		setStatus(`Recycled ${recycled} weak splats into high-error motion support.`);
	}
	renderOnce();
	await readLossIfReady(iterations === 1);
	if (iterations === 1) {
		await readGridLossIfReady(true);
	} else {
		void readGridLossIfReady(false);
	}
}

async function frameLoop(nowMs) {
	try {
		const deltaMs = Math.max(1, nowMs - lastFrameMs);
		lastFrameMs = nowMs;
		fpsValue.textContent = `${Math.round(1000 / deltaMs)} fps`;
		advancePreviewTime(deltaMs);

		if (!booting && running && trainer && !gridLossBusy) {
			await stepTrainer(trainer.splatCount >= 320 ? 1 : 2);
		} else if (!booting) {
			renderOnce();
		}
		if (!booting && trainer && nowMs - lastRateMs >= 1000) {
			const stepDelta = trainer.stepCount - lastRateStep;
			const seconds = (nowMs - lastRateMs) / 1000;
			stepRateValue.textContent = `${(stepDelta / seconds).toFixed(1)}`;
			lastRateStep = trainer.stepCount;
			lastRateMs = nowMs;
		}
	} catch (error) {
		console.warn("Frame loop recovered after render/train failure.", error);
		setRunning(false);
		setStatus(error instanceof Error ? error.message : String(error));
	} finally {
		animationHandle = requestAnimationFrame((nextMs) => {
			void frameLoop(nextMs);
		});
	}
}

runButton.addEventListener("click", () => {
	setRunning(!running);
});

stepButton.addEventListener("click", () => {
	void stepTrainer(1);
});

resetButton.addEventListener("click", () => {
	void resetTrainer();
});

modeSelect.addEventListener("change", () => {
	if (dataset && !booting) {
		void resetTrainer();
	}
});

splatSlider.addEventListener("input", updateSliderLabels);
splatSlider.addEventListener("change", () => {
	if (dataset && !booting) {
		void resetTrainer();
	}
});

timeSlider.addEventListener("input", () => {
	updateSliderLabels();
	updateTargetCanvas();
	renderOnce();
});
timeLoopToggle.addEventListener("change", updateSliderLabels);
timeSpeedSlider.addEventListener("input", updateSliderLabels);

targetViewSelect?.addEventListener("change", () => {
	lastErrorMapKey = "";
	updateTargetCanvas();
});
resultViewSelect?.addEventListener("change", renderOnce);

temporalSlider.addEventListener("input", () => {
	updateSliderLabels();
	renderOnce();
});

lrSlider.addEventListener("input", updateSliderLabels);
samplesSlider.addEventListener("input", updateSliderLabels);
motionMixSlider.addEventListener("input", updateSliderLabels);
motionMixSlider.addEventListener("change", updateSliderLabels);
staticMixSlider.addEventListener("input", updateSliderLabels);
staticMixSlider.addEventListener("change", updateSliderLabels);
supportGuardSlider.addEventListener("input", updateSliderLabels);
supportGuardSlider.addEventListener("change", updateSliderLabels);

new ResizeObserver(() => {
	renderOnce();
}).observe(renderCanvas);

window.addEventListener("beforeunload", () => {
	if (animationHandle) {
		cancelAnimationFrame(animationHandle);
	}
	trainer?.dispose();
});

void boot().then(() => {
	animationHandle = requestAnimationFrame((nowMs) => {
		lastFrameMs = nowMs;
		void frameLoop(nowMs);
	});
});
