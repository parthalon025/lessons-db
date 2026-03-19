// API client for lessons-db FastAPI backend.
// All endpoints are relative — the SPA is served from the same origin.

const BASE = '/api';

/**
 * Safe JSON fetch — checks Content-Type before parsing.
 * Returns parsed JSON on success, throws on HTTP error.
 */
async function fetchJSON(url, opts = {}) {
  const resp = await fetch(url, opts);
  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  const ct = resp.headers.get('content-type') || '';
  if (!ct.includes('application/json')) {
    throw new Error(`Expected JSON, got ${ct}`);
  }
  return resp.json();
}

// ── Lessons ──

export function fetchLessons(params = {}) {
  const qs = new URLSearchParams();
  if (params.q) qs.set('q', params.q);
  if (params.category) qs.set('category', params.category);
  if (params.tier) qs.set('tier', params.tier);
  if (params.polarity) qs.set('polarity', params.polarity);
  if (params.sort) qs.set('sort', params.sort);
  if (params.limit) qs.set('limit', String(params.limit));
  if (params.offset) qs.set('offset', String(params.offset));
  const query = qs.toString();
  return fetchJSON(`${BASE}/lessons${query ? '?' + query : ''}`);
}

export function fetchLessonById(id) {
  return fetchJSON(`${BASE}/lessons/${id}`);
}

export function fetchLessonsStats() {
  return fetchJSON(`${BASE}/lessons/stats`);
}

export function fetchCategories() {
  return fetchJSON(`${BASE}/lessons/categories`);
}

// ── Scan health ──

export function fetchScanSummary() {
  return fetchJSON(`${BASE}/scan/summary`);
}

// ── Capture drafts ──

export function fetchCaptureDrafts(status = 'pending') {
  return fetchJSON(`${BASE}/capture-drafts?status=${status}`);
}

export function patchCaptureDraft(id, status) {
  return fetchJSON(`${BASE}/capture-drafts/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
}

// ── Fix queue ──

export function fetchFixQueue(status = 'pending') {
  return fetchJSON(`${BASE}/fix-queue?status=${status}`);
}

export function patchFixQueue(id, status) {
  return fetchJSON(`${BASE}/fix-queue/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ status }),
  });
}

// ── Prevention ──

export function fetchPreventionReport(windowDays = 30) {
  return fetchJSON(`${BASE}/prevention/report?window_days=${windowDays}`);
}

export function fetchRecurrence(windowDays = 7, threshold = 2) {
  return fetchJSON(`${BASE}/prevention/recurrence?window_days=${windowDays}&threshold=${threshold}`);
}

// ── Mining ──

export function fetchMiningHistory(limit = 20) {
  return fetchJSON(`${BASE}/mining/history?limit=${limit}`);
}

// ── Calibration ──

export function fetchCalibrationHistory(limit = 20) {
  return fetchJSON(`${BASE}/calibration/history?limit=${limit}`);
}
