// Lessons page — searchable, filterable lesson table.
// What it shows: All lessons with filters by text, category, tier, polarity, and sort order.
// Decision it drives: "Find a specific lesson. Review lesson details. Understand the DB contents."

import { useEffect } from 'preact/hooks';
import {
  lessons, lessonsError, refreshLessons,
  categories, refreshCategories,
  filterQuery, filterCategory, filterTier, filterPolarity, filterSort, filterOffset,
} from '../stores/lessons.js';
import LoadingState from '../components/LoadingState.jsx';

export default function Lessons() {
  useEffect(() => {
    refreshCategories();
    refreshLessons();
  }, []);

  const data = lessons.value;
  const cats = categories.value;

  function applyFilters() {
    filterOffset.value = 0;
    refreshLessons();
  }

  function nextPage() {
    filterOffset.value = filterOffset.value + 50;
    refreshLessons();
  }

  function prevPage() {
    filterOffset.value = Math.max(0, filterOffset.value - 50);
    refreshLessons();
  }

  return (
    <div>
      <div class="page-header">
        <div class="page-title">LESSONS</div>
        <div class="page-subtitle">BROWSE AND SEARCH</div>
      </div>

      {/* Filter bar */}
      <div class="filter-bar">
        <input
          type="text"
          placeholder="SEARCH..."
          value={filterQuery.value}
          onInput={(ev) => { filterQuery.value = ev.target.value; }}
          onKeyDown={(ev) => { if (ev.key === 'Enter') applyFilters(); }}
        />
        <select
          value={filterCategory.value}
          onChange={(ev) => { filterCategory.value = ev.target.value; applyFilters(); }}
        >
          <option value="">ALL CATEGORIES</option>
          {cats.map((cat) => <option key={cat} value={cat}>{cat.toUpperCase()}</option>)}
        </select>
        <select
          value={filterTier.value}
          onChange={(ev) => { filterTier.value = ev.target.value; applyFilters(); }}
        >
          <option value="">ALL TIERS</option>
          {['S', 'A', 'B', 'C', 'D'].map((tier) => <option key={tier} value={tier}>{tier}</option>)}
        </select>
        <select
          value={filterPolarity.value}
          onChange={(ev) => { filterPolarity.value = ev.target.value; applyFilters(); }}
        >
          <option value="">ALL POLARITY</option>
          <option value="positive">POSITIVE</option>
          <option value="negative">NEGATIVE</option>
        </select>
        <select
          value={filterSort.value}
          onChange={(ev) => { filterSort.value = ev.target.value; applyFilters(); }}
        >
          <option value="">NEWEST FIRST</option>
          <option value="id_asc">OLDEST FIRST</option>
          <option value="severity">BY SEVERITY</option>
          <option value="recurrence_count">BY RECURRENCE</option>
        </select>
        <button class="btn" onClick={applyFilters}>SEARCH</button>
      </div>

      {/* Results */}
      {!data.lessons ? <LoadingState /> : (
        <div>
          <div style="margin-bottom: 8px; font-size: 0.75rem; color: var(--text-muted);">
            {data.total} TOTAL — SHOWING {data.offset + 1}–{Math.min(data.offset + 50, data.total)}
          </div>

          <table class="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>TITLE</th>
                <th>CATEGORY</th>
                <th>TIER</th>
                <th>POLARITY</th>
              </tr>
            </thead>
            <tbody>
              {data.lessons.map((lesson) => (
                <tr key={lesson.id}>
                  <td>{lesson.id}</td>
                  <td>{lesson.title || lesson.one_liner || '—'}</td>
                  <td>{lesson.category || '—'}</td>
                  <td>{lesson.tier || '—'}</td>
                  <td>
                    <span class={lesson.polarity === 'positive' ? 'status-ok' : 'status-alert'}>
                      {(lesson.polarity || '—').toUpperCase()}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>

          {/* Pagination */}
          <div style="display: flex; gap: 8px; margin-top: 16px; justify-content: center;">
            {data.offset > 0 && (
              <button class="btn" onClick={prevPage}>PREV</button>
            )}
            {data.offset + 50 < data.total && (
              <button class="btn" onClick={nextPage}>NEXT</button>
            )}
          </div>
        </div>
      )}

      {lessonsError.value && (
        <div class="status-alert">{lessonsError.value}</div>
      )}
    </div>
  );
}
