export const BROWSER_ADAM_BETA1 = 0.9;
export const BROWSER_ADAM_BETA2 = 0.999;
export const BROWSER_ADAM_EPSILON = 1e-8;
export const DENSITY_STAT_DECAY = 0.999;
export const LEARNING_RATE_DECAY_STEPS = 120000;

function clamp(value, minimum, maximum) {
	return Math.min(maximum, Math.max(minimum, value));
}

export function learningRateMultipliers(step, enabled = true) {
	if (!Number.isFinite(step) || step < 0) {
		throw new RangeError("step must be finite and non-negative.");
	}
	if (!enabled) return { geometry: 1, appearance: 1, progress: 0 };
	const progress = clamp(step / LEARNING_RATE_DECAY_STEPS, 0, 1);
	return {
		geometry: 10 ** (-2 * progress),
		appearance: 10 ** (-progress),
		progress,
	};
}

export function browserLearningRates(baseScale, step, decayEnabled = true) {
	if (!Number.isFinite(baseScale) || baseScale < 0) {
		throw new RangeError("baseScale must be finite and non-negative.");
	}
	const multipliers = learningRateMultipliers(step, decayEnabled);
	return {
		position: baseScale * 0.00035 * multipliers.geometry,
		motion: baseScale * 0.0002 * multipliers.geometry,
		color: baseScale * 0.0015 * multipliers.appearance,
		opacity: baseScale * 0.0008 * multipliers.appearance,
		...multipliers,
	};
}
