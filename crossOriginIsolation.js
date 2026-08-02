export async function ensureStaticHostIsolation(scope = globalThis) {
	if (scope.crossOriginIsolated || !scope.navigator?.serviceWorker) return false;
	const registration = await scope.navigator.serviceWorker.register("./coi-serviceworker.js", { scope: "./" });
	if (!scope.navigator.serviceWorker.controller) {
		await new Promise((resolve) => {
			const worker = registration.installing ?? registration.waiting ?? registration.active;
			if (!worker || worker.state === "activated") resolve();
			else worker.addEventListener("statechange", () => {
				if (worker.state === "activated") resolve();
			});
		});
		scope.location.reload();
		return true;
	}
	return false;
}
