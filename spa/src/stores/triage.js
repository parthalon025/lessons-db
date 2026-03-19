// Triage store — capture drafts and fix queue items pending action.
// What it shows: Pending items that need human decision (approve/dismiss drafts, apply/skip fixes).
// Decision it drives: "What needs my attention right now? Approve or dismiss each item."

import { signal } from '@preact/signals';
import {
  fetchCaptureDrafts,
  patchCaptureDraft,
  fetchFixQueue,
  patchFixQueue,
} from '../api.js';

export const drafts = signal([]);
export const draftsError = signal(null);
export const fixes = signal([]);
export const fixesError = signal(null);

export async function refreshDrafts() {
  try {
    drafts.value = await fetchCaptureDrafts('pending');
    draftsError.value = null;
  } catch (err) {
    draftsError.value = err.message;
  }
}

export async function refreshFixes() {
  try {
    fixes.value = await fetchFixQueue('pending');
    fixesError.value = null;
  } catch (err) {
    fixesError.value = err.message;
  }
}

export async function approveDraft(id) {
  await patchCaptureDraft(id, 'promoted');
  await refreshDrafts();
}

export async function dismissDraft(id) {
  await patchCaptureDraft(id, 'dismissed');
  await refreshDrafts();
}

export async function applyFix(id) {
  await patchFixQueue(id, 'applied');
  await refreshFixes();
}

export async function skipFix(id) {
  await patchFixQueue(id, 'skipped');
  await refreshFixes();
}
