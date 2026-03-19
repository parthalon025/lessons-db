// Pipelines store — mining and calibration run history.
// What it shows: Recent mining runs (GitHub lesson mining) and calibration runs (BugsInPy pipeline).
// Decision it drives: "When did the last mining/calibration run? Were there errors?"

import { signal } from '@preact/signals';
import { fetchMiningHistory, fetchCalibrationHistory } from '../api.js';

export const miningHistory = signal([]);
export const miningError = signal(null);
export const calibrationHistory = signal([]);
export const calibrationError = signal(null);

export async function refreshMining() {
  try {
    miningHistory.value = await fetchMiningHistory(20);
    miningError.value = null;
  } catch (err) {
    miningError.value = err.message;
  }
}

export async function refreshCalibration() {
  try {
    calibrationHistory.value = await fetchCalibrationHistory(20);
    calibrationError.value = null;
  } catch (err) {
    calibrationError.value = err.message;
  }
}
