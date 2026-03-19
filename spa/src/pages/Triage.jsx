// Triage page — pending capture drafts and fix queue items.
// What it shows: Items awaiting human decision — drafts to approve/dismiss, fixes to apply/skip.
// Decision it drives: "Review each pending item and take action. Clear the inbox."

import { useEffect } from 'preact/hooks';
import {
  drafts, draftsError, refreshDrafts, approveDraft, dismissDraft,
  fixes, fixesError, refreshFixes, applyFix, skipFix,
} from '../stores/triage.js';
import { useActionFeedback } from '../hooks/useActionFeedback.js';
import { useShatter } from '../hooks/useShatter.js';
import LoadingState from '../components/LoadingState.jsx';

export default function Triage() {
  useEffect(() => {
    refreshDrafts();
    refreshFixes();
  }, []);

  const draftList = drafts.value;
  const fixList = fixes.value;

  return (
    <div>
      <div class="page-header">
        <div class="page-title">TRIAGE</div>
        <div class="page-subtitle">PENDING ACTIONS</div>
      </div>

      {/* Capture drafts */}
      <div class="section">
        <div class="section-title">
          CAPTURE DRAFTS ({draftList.length} PENDING)
        </div>
        {draftList.length === 0 ? (
          <div class="loading-container">NO PENDING DRAFTS</div>
        ) : (
          draftList.map((draft) => (
            <DraftCard key={draft.id} draft={draft} />
          ))
        )}
        {draftsError.value && (
          <div class="status-alert">{draftsError.value}</div>
        )}
      </div>

      {/* Fix queue */}
      <div class="section">
        <div class="section-title">
          FIX QUEUE ({fixList.length} PENDING)
        </div>
        {fixList.length === 0 ? (
          <div class="loading-container">NO PENDING FIXES</div>
        ) : (
          fixList.map((fix) => (
            <FixCard key={fix.id} fix={fix} />
          ))
        )}
        {fixesError.value && (
          <div class="status-alert">{fixesError.value}</div>
        )}
      </div>
    </div>
  );
}

function DraftCard({ draft }) {
  const [fb, act] = useActionFeedback();
  const [shatterRef, shatterFire] = useShatter('earned');

  function handleApprove() {
    act('PROMOTING...', () => approveDraft(draft.id), () => 'PROMOTED');
  }

  function handleDismiss() {
    shatterFire();
    act('DISMISSING...', () => dismissDraft(draft.id), () => 'DISMISSED');
  }

  return (
    <div class="triage-card" ref={shatterRef}>
      <div class="triage-card-title">{draft.title || draft.one_liner || `DRAFT #${draft.id}`}</div>
      {draft.description && (
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px;">
          {draft.description}
        </div>
      )}
      {draft.source && (
        <div style="font-size: 0.7rem; color: var(--text-muted);">
          SOURCE: {draft.source || draft.detection_source || '—'}
        </div>
      )}
      <div class="triage-actions">
        <button class="btn btn-phosphor" onClick={handleApprove} disabled={fb.phase === 'loading'}>
          APPROVE
        </button>
        <button class="btn btn-threat" onClick={handleDismiss} disabled={fb.phase === 'loading'}>
          DISMISS
        </button>
      </div>
      {fb.msg && (
        <div class={`status-${fb.phase === 'success' ? 'ok' : fb.phase === 'error' ? 'alert' : 'warn'}`}
             style="margin-top: 4px; font-size: 0.75rem;">
          {fb.msg}
        </div>
      )}
    </div>
  );
}

function FixCard({ fix }) {
  const [fb, act] = useActionFeedback();
  const [shatterRef, shatterFire] = useShatter('complete');

  function handleApply() {
    shatterFire();
    act('APPLYING...', () => applyFix(fix.id), () => 'APPLIED');
  }

  function handleSkip() {
    act('SKIPPING...', () => skipFix(fix.id), () => 'SKIPPED');
  }

  return (
    <div class="triage-card" ref={shatterRef}>
      <div class="triage-card-title">{fix.title || fix.description || `FIX #${fix.id}`}</div>
      {fix.lesson_id && (
        <div style="font-size: 0.7rem; color: var(--text-muted);">
          LESSON #{fix.lesson_id} — SEVERITY: {fix.severity || '?'}
        </div>
      )}
      <div class="triage-actions">
        <button class="btn btn-phosphor" onClick={handleApply} disabled={fb.phase === 'loading'}>
          APPLY
        </button>
        <button class="btn" onClick={handleSkip} disabled={fb.phase === 'loading'}>
          SKIP
        </button>
      </div>
      {fb.msg && (
        <div class={`status-${fb.phase === 'success' ? 'ok' : fb.phase === 'error' ? 'alert' : 'warn'}`}
             style="margin-top: 4px; font-size: 0.75rem;">
          {fb.msg}
        </div>
      )}
    </div>
  );
}
