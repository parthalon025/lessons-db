// Prevention store — effectiveness report and high-velocity recurrence data.
// What it shows: 30-day prevention effectiveness + lessons with high recurrence velocity.
// Decision it drives: "Is the prevention pipeline catching real issues? Which lessons recur most?"

import { signal } from '@preact/signals';
import { fetchPreventionReport, fetchRecurrence } from '../api.js';

export const preventionReport = signal(null);
export const preventionError = signal(null);
export const recurrence = signal([]);
export const recurrenceError = signal(null);

export async function refreshPrevention() {
  try {
    preventionReport.value = await fetchPreventionReport(30);
    preventionError.value = null;
  } catch (err) {
    preventionError.value = err.message;
  }
}

export async function refreshRecurrence() {
  try {
    recurrence.value = await fetchRecurrence(7, 2);
    recurrenceError.value = null;
  } catch (err) {
    recurrenceError.value = err.message;
  }
}
