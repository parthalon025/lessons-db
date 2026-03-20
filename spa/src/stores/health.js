// Health store — scan pipeline health metrics from /api/scan/summary.
// What it shows: 6 health indicators (promotion rate, drafts captured, sessions, scan age, embed failures, FSRS backlog).
// Decision it drives: "Is the nightly pipeline working? Do I need to intervene?"

import { signal } from '@preact/signals';
import { setFacilityState } from 'superhot-ui';
import { fetchScanSummary } from '../api.js';

export const healthMetrics = signal(null);
export const healthError = signal(null);

/** Map the worst per-metric status to a facility state. */
function syncFacilityState(metrics) {
  if (!metrics) return;
  const statuses = Object.values(metrics).map(m => m.status);
  if (statuses.includes('alert')) setFacilityState('breach');
  else if (statuses.includes('warn')) setFacilityState('alert');
  else setFacilityState('normal');
}

export async function refreshHealth() {
  try {
    const data = await fetchScanSummary();
    healthMetrics.value = data;
    healthError.value = null;
    syncFacilityState(data);
  } catch (err) {
    healthError.value = err.message;
  }
}
