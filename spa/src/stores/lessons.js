// Lessons store — paginated lesson list with filtering.
// What it shows: Searchable, filterable table of all lessons.
// Decision it drives: "Which lessons exist? Filter by category/tier/polarity to find relevant ones."

import { signal } from '@preact/signals';
import { fetchLessons, fetchCategories } from '../api.js';

export const lessons = signal({ lessons: [], total: 0, offset: 0 });
export const lessonsError = signal(null);
export const categories = signal([]);

// Filter state
export const filterQuery = signal('');
export const filterCategory = signal('');
export const filterTier = signal('');
export const filterPolarity = signal('');
export const filterSort = signal('');
export const filterOffset = signal(0);

export async function refreshLessons() {
  try {
    const data = await fetchLessons({
      q: filterQuery.value || undefined,
      category: filterCategory.value || undefined,
      tier: filterTier.value || undefined,
      polarity: filterPolarity.value || undefined,
      sort: filterSort.value || undefined,
      offset: filterOffset.value,
      limit: 50,
    });
    lessons.value = data;
    lessonsError.value = null;
  } catch (err) {
    lessonsError.value = err.message;
  }
}

export async function refreshCategories() {
  try {
    categories.value = await fetchCategories();
  } catch (err) {
    console.warn('[categories]', err.message);
  }
}
