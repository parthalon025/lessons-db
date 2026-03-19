// useActionFeedback hook — loading/success/error feedback for action buttons.
// Pattern: const [fb, act] = useActionFeedback();
//   <button disabled={fb.phase==='loading'} onClick={() => act('PROCESSING...', asyncFn)}>
//   {fb.msg && <div class={`action-fb action-fb--${fb.phase}`}>{fb.msg}</div>}

import { useState, useCallback, useRef } from 'preact/hooks';

/**
 * @returns {[{phase: string, msg: string}, Function]}
 */
export function useActionFeedback() {
  const [state, setState] = useState({ phase: 'idle', msg: '' });
  const timerRef = useRef(null);
  const loadingRef = useRef(false);

  const run = useCallback(async (loadingMsg, fn, successFn) => {
    // Double-click guard — ref-based to avoid stale closure over state.phase
    if (loadingRef.current) return;
    loadingRef.current = true;

    if (timerRef.current) clearTimeout(timerRef.current);

    setState({ phase: 'loading', msg: loadingMsg });

    try {
      const result = await fn();
      const successMsg = successFn ? successFn(result) : 'DONE';
      setState({ phase: 'success', msg: successMsg });
    } catch (err) {
      setState({ phase: 'error', msg: err.message || 'FAILED' });
    } finally {
      loadingRef.current = false;
    }

    timerRef.current = setTimeout(() => {
      setState({ phase: 'idle', msg: '' });
    }, 2000);
  }, []);

  return [state, run];
}
