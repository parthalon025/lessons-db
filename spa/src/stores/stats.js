// Stats store — lesson counts, polarity breakdown, tier distribution.
// What it shows: Top-level KPI numbers (total, positive, negative, drafts pending).
// Decision it drives: "Is the system capturing and triaging lessons at a healthy rate?"

import { signal } from '@preact/signals';
import { fetchLessonsStats } from '../api.js';

export const stats = signal(null);
export const statsError = signal(null);

export async function refreshStats() {
  try {
    stats.value = await fetchLessonsStats();
    statsError.value = null;
  } catch (err) {
    statsError.value = err.message;
  }
}
