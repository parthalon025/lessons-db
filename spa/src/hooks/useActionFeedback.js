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

  const run = useCallback(async (loadingMsg, fn, successFn) => {
    // Double-click guard
    if (state.phase === 'loading') return;

    if (timerRef.current) clearTimeout(timerRef.current);

    setState({ phase: 'loading', msg: loadingMsg });

    try {
      const result = await fn();
      const successMsg = successFn ? successFn(result) : 'DONE';
      setState({ phase: 'success', msg: successMsg });
    } catch (err) {
      setState({ phase: 'error', msg: err.message || 'FAILED' });
    }

    timerRef.current = setTimeout(() => {
      setState({ phase: 'idle', msg: '' });
    }, 2000);
  }, [state.phase]);

  return [state, run];
}
