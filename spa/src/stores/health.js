// Health store — scan pipeline health metrics from /api/scan/summary.
// What it shows: 6 health indicators (promotion rate, drafts captured, sessions, scan age, embed failures, FSRS backlog).
// Decision it drives: "Is the nightly pipeline working? Do I need to intervene?"

import { signal } from '@preact/signals';
import { fetchScanSummary } from '../api.js';

export const healthMetrics = signal(null);
export const healthError = signal(null);

export async function refreshHealth() {
  try {
    healthMetrics.value = await fetchScanSummary();
    healthError.value = null;
  } catch (err) {
    healthError.value = err.message;
  }
}
