// useShatter hook — tiered shatter effect for action buttons.
// 3 tiers: earned (7 fragments), complete (6), routine (3).
// Integrates with superhot-ui shatterElement if available.

import { useRef, useCallback } from 'preact/hooks';

const TIER_FRAGMENTS = {
  earned: 7,
  complete: 6,
  routine: 3,
};

/**
 * @param {'earned'|'complete'|'routine'} tier
 * @returns {[import('preact').RefObject, () => void]}
 */
export function useShatter(tier = 'routine') {
  const ref = useRef(null);

  const fire = useCallback(() => {
    const el = ref.current;
    if (!el) return;

    const fragments = TIER_FRAGMENTS[tier] || 3;

    // Try to use superhot-ui shatterElement if available
    if (typeof window !== 'undefined' && window.__shatterElement) {
      window.__shatterElement(el, { fragments });
      return;
    }

    // Lightweight fallback — brief scale pulse
    el.style.transition = 'transform 0.15s ease-out';
    el.style.transform = 'scale(0.95)';
    requestAnimationFrame(() => {
      setTimeout(() => {
        el.style.transform = 'scale(1)';
      }, 150);
    });
  }, [tier]);

  return [ref, fire];
}
