// Polling manager — starts/stops intervals for dashboard data refresh.
// Each poller has an id and interval; duplicate ids are prevented.

const _pollers = new Map();

/**
 * Start a polling interval. If one with the same id exists, it is replaced.
 *
 * @param {string} id - Unique identifier for this poller
 * @param {Function} fn - Async function to call on each tick
 * @param {number} intervalMs - Milliseconds between ticks
 * @param {boolean} [immediate=true] - Call fn immediately on start
 */
export function startPolling(id, fn, intervalMs, immediate = true) {
  stopPolling(id);
  if (immediate) {
    try { fn(); } catch (_) { /* swallow initial error */ }
  }
  const timerId = setInterval(() => {
    try { fn(); } catch (_) { /* swallow poll error */ }
  }, intervalMs);
  _pollers.set(id, timerId);
}

/**
 * Stop a poller by id.
 * @param {string} id
 */
export function stopPolling(id) {
  const timerId = _pollers.get(id);
  if (timerId !== undefined) {
    clearInterval(timerId);
    _pollers.delete(id);
  }
}

/**
 * Stop all active pollers.
 */
export function stopAllPolling() {
  for (const [id, timerId] of _pollers) {
    clearInterval(timerId);
  }
  _pollers.clear();
}
