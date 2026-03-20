// useShatter hook — tiered shatter effect for action buttons.
// 3 tiers: earned (7 fragments), complete (6), routine (3).
// Uses superhot-ui shatterElement directly.

import { useRef, useCallback } from 'preact/hooks';
import { shatterElement } from 'superhot-ui';

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
    shatterElement(el, { fragments });
  }, [tier]);

  return [ref, fire];
}
