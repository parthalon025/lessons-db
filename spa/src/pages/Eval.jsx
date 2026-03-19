// Eval page — prevention effectiveness and pipeline run history.
// What it shows: 30-day prevention report, high-recurrence lessons, mining/calibration run history.
// Decision it drives: "Is prevention working? Which lessons recur most? When did pipelines last run?"

import { useEffect } from 'preact/hooks';
import {
  preventionReport, preventionError, refreshPrevention,
  recurrence, recurrenceError, refreshRecurrence,
} from '../stores/prevention.js';
import {
  miningHistory, miningError, refreshMining,
  calibrationHistory, calibrationError, refreshCalibration,
} from '../stores/pipelines.js';
import LoadingState from '../components/LoadingState.jsx';

export default function Eval() {
  useEffect(() => {
    refreshPrevention();
    refreshRecurrence();
    refreshMining();
    refreshCalibration();
  }, []);

  const report = preventionReport.value;
  const recurrenceList = recurrence.value;
  const mining = miningHistory.value;
  const calibration = calibrationHistory.value;

  return (
    <div>
      <div class="page-header">
        <div class="page-title">EVAL</div>
        <div class="page-subtitle">PREVENTION + PIPELINES</div>
      </div>

      {/* Prevention report */}
      <div class="section">
        <div class="section-title">PREVENTION EFFECTIVENESS (30 DAYS)</div>
        {!report ? <LoadingState message="LOADING REPORT..." /> : (
          <div class="stat-grid">
            {Object.entries(report).map(([key, val]) => (
              <div class="stat-card" key={key}>
                <div class="stat-card-label">{key.replace(/_/g, ' ').toUpperCase()}</div>
                <div class="stat-card-value">
                  {typeof val === 'number' ? val : typeof val === 'object' ? JSON.stringify(val) : String(val)}
                </div>
              </div>
            ))}
          </div>
        )}
        {preventionError.value && <div class="status-alert">{preventionError.value}</div>}
      </div>

      {/* High recurrence */}
      <div class="section">
        <div class="section-title">HIGH RECURRENCE (7 DAYS)</div>
        {recurrenceList.length === 0 ? (
          <div class="loading-container">NO HIGH-VELOCITY LESSONS</div>
        ) : (
          <table class="data-table">
            <thead>
              <tr>
                <th>LESSON ID</th>
                <th>TITLE</th>
                <th>RECURRENCE</th>
              </tr>
            </thead>
            <tbody>
              {recurrenceList.map((item) => (
                <tr key={item.id || item.lesson_id}>
                  <td>{item.id || item.lesson_id}</td>
                  <td>{item.title || item.one_liner || '—'}</td>
                  <td class="status-warn">{item.velocity || item.count || '?'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {recurrenceError.value && <div class="status-alert">{recurrenceError.value}</div>}
      </div>

      {/* Mining history */}
      <div class="section">
        <div class="section-title">MINING RUNS</div>
        {mining.length === 0 ? (
          <div class="loading-container">NO MINING RUNS</div>
        ) : (
          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>DATE</th>
                <th>REPOS</th>
                <th>COMMITS</th>
                <th>DRAFTED</th>
                <th>ERRORS</th>
              </tr>
            </thead>
            <tbody>
              {mining.slice(0, 10).map((run) => (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.run_date || '—'}</td>
                  <td>{run.repos_searched || 0}</td>
                  <td>{run.commits_analyzed || 0}</td>
                  <td>{run.drafted || 0}</td>
                  <td class={run.error_count > 0 ? 'status-alert' : ''}>
                    {run.error_count || 0}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {miningError.value && <div class="status-alert">{miningError.value}</div>}
      </div>

      {/* Calibration history */}
      <div class="section">
        <div class="section-title">CALIBRATION RUNS</div>
        {calibration.length === 0 ? (
          <div class="loading-container">NO CALIBRATION RUNS</div>
        ) : (
          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>DATE</th>
                <th>BUGS SAMPLED</th>
                <th>PASS RATE</th>
                <th>GATE 14</th>
              </tr>
            </thead>
            <tbody>
              {calibration.slice(0, 10).map((run) => (
                <tr key={run.id}>
                  <td>{run.id}</td>
                  <td>{run.run_date || '—'}</td>
                  <td>{run.bugs_sampled || 0}</td>
                  <td>{run.pass_rate != null ? `${(run.pass_rate * 100).toFixed(1)}%` : '—'}</td>
                  <td class={run.gate14_pass ? 'status-ok' : 'status-alert'}>
                    {run.gate14_pass ? 'PASS' : 'FAIL'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {calibrationError.value && <div class="status-alert">{calibrationError.value}</div>}
      </div>
    </div>
  );
}
