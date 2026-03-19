// Dashboard page — system overview with KPI stats + health indicators.
// What it shows: Total lessons, polarity breakdown, pending drafts, and 6 pipeline health metrics.
// Decision it drives: "Is the system healthy? Do I need to triage drafts or investigate pipeline issues?"

import { useEffect } from 'preact/hooks';
import { stats, statsError, refreshStats } from '../stores/stats.js';
import { healthMetrics, healthError, refreshHealth } from '../stores/health.js';
import { startPolling, stopPolling } from '../polling.js';
import LoadingState from '../components/LoadingState.jsx';

const POLL_ID = 'dashboard';
const POLL_INTERVAL = 30000; // 30s

export default function Dashboard() {
  useEffect(() => {
    refreshStats();
    refreshHealth();
    startPolling(POLL_ID, () => { refreshStats(); refreshHealth(); }, POLL_INTERVAL, false);
    return () => stopPolling(POLL_ID);
  }, []);

  const s = stats.value;
  const hm = healthMetrics.value;

  return (
    <div>
      <div class="page-header">
        <div class="page-title">DASHBOARD</div>
        <div class="page-subtitle">SYSTEM OVERVIEW</div>
      </div>

      {/* KPI stat cards */}
      <div class="section">
        <div class="section-title">LESSON METRICS</div>
        {!s ? <LoadingState message="LOADING STATS..." /> : (
          <div class="stat-grid">
            <div class="stat-card">
              <div class="stat-card-label">TOTAL LESSONS</div>
              <div class="stat-card-value">{s.total}</div>
            </div>
            <div class="stat-card">
              <div class="stat-card-label">POSITIVE</div>
              <div class="stat-card-value status-ok">{s.positive}</div>
            </div>
            <div class="stat-card">
              <div class="stat-card-label">NEGATIVE</div>
              <div class="stat-card-value status-alert">{s.negative}</div>
            </div>
            <div class="stat-card">
              <div class="stat-card-label">PENDING DRAFTS</div>
              <div class="stat-card-value">{s.pending_drafts > 0 ? (
                <span class="status-warn">{s.pending_drafts}</span>
              ) : '0'}</div>
            </div>
          </div>
        )}
        {statsError.value && (
          <div class="status-alert">{statsError.value}</div>
        )}
      </div>

      {/* Top tiers */}
      {s && s.top_tiers && s.top_tiers.length > 0 && (
        <div class="section">
          <div class="section-title">TOP TIERS</div>
          <div class="stat-grid">
            {s.top_tiers.map((tier) => (
              <div class="stat-card" key={tier.tier}>
                <div class="stat-card-label">{tier.tier || 'UNSET'}</div>
                <div class="stat-card-value">{tier.count}</div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Health metrics */}
      <div class="section">
        <div class="section-title">PIPELINE HEALTH</div>
        {!hm ? <LoadingState message="LOADING HEALTH..." /> : (
          <div class="health-grid">
            {Object.entries(hm).map(([key, metric]) => (
              <HealthCard key={key} name={key} metric={metric} />
            ))}
          </div>
        )}
        {healthError.value && (
          <div class="status-alert">{healthError.value}</div>
        )}
      </div>
    </div>
  );
}

function HealthCard({ name, metric }) {
  const statusClass = metric.status === 'ok' ? 'status-ok'
    : metric.status === 'warn' ? 'status-warn'
    : 'status-alert';

  return (
    <div class="health-card">
      <div class="health-card-header">
        <span class="health-card-label">{formatMetricName(name)}</span>
        <span class={statusClass}>{metric.status ? metric.status.toUpperCase() : '?'}</span>
      </div>
      <div class="health-card-value">{metric.label || '—'}</div>
      {metric.decision_context && (
        <div class="health-card-context">{metric.decision_context}</div>
      )}
    </div>
  );
}

function formatMetricName(name) {
  return name.replace(/_/g, ' ').toUpperCase();
}
