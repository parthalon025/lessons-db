// Admin page — system administration: category list, trigger actions.
// What it shows: Lesson categories, and controls to trigger mining/calibration/scan runs.
// Decision it drives: "Run a mining or calibration pipeline. View system metadata."

import { useEffect } from 'preact/hooks';
import { categories, refreshCategories } from '../stores/lessons.js';
import { useActionFeedback } from '../hooks/useActionFeedback.js';

export default function Admin() {
  useEffect(() => {
    refreshCategories();
  }, []);

  const cats = categories.value;

  return (
    <div>
      <div class="page-header">
        <div class="page-title">ADMIN</div>
        <div class="page-subtitle">SYSTEM CONTROLS</div>
      </div>

      {/* Trigger actions */}
      <div class="section">
        <div class="section-title">PIPELINE TRIGGERS</div>
        <div style="display: flex; gap: 16px; flex-wrap: wrap;">
          <TriggerButton
            label="TRIGGER MINING RUN"
            endpoint="/api/mining/run"
          />
          <TriggerButton
            label="TRIGGER CALIBRATION"
            endpoint="/api/calibration/run"
          />
          <TriggerButton
            label="TRIGGER SECURITY SCAN"
            endpoint="/api/security/scan"
          />
          <TriggerButton
            label="POPULATE FIX QUEUE"
            endpoint="/api/fix-queue/populate"
          />
        </div>
      </div>

      {/* Categories */}
      <div class="section">
        <div class="section-title">CATEGORIES ({cats.length})</div>
        {cats.length === 0 ? (
          <div class="loading-container">NO CATEGORIES</div>
        ) : (
          <div style="display: flex; flex-wrap: wrap; gap: 8px;">
            {cats.map((cat) => (
              <span key={cat} class="stat-card" style="padding: 4px 12px; font-size: 0.8rem;">
                {cat.toUpperCase()}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TriggerButton({ label, endpoint }) {
  const [fb, act] = useActionFeedback();

  function handleClick() {
    act('SUBMITTING...', async () => {
      const resp = await fetch(endpoint, { method: 'POST' });
      if (!resp.ok) {
        const text = await resp.text().catch(() => '');
        throw new Error(`${resp.status}: ${text}`);
      }
      return resp.json();
    }, () => 'QUEUED');
  }

  return (
    <div>
      <button class="btn btn-phosphor" onClick={handleClick} disabled={fb.phase === 'loading'}>
        {fb.phase === 'loading' ? fb.msg : label}
      </button>
      {fb.msg && fb.phase !== 'loading' && (
        <div class={`status-${fb.phase === 'success' ? 'ok' : 'alert'}`}
             style="font-size: 0.7rem; margin-top: 4px;">
          {fb.msg}
        </div>
      )}
    </div>
  );
}
