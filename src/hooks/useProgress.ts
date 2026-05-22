import { useState, useEffect } from 'react';
import type { ProgressEvent } from '../types';

export function useProgress() {
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    let unlisten: (() => void) | undefined;

    async function setup() {
      try {
        const { listen } = await import('@tauri-apps/api/event');
        const unlistenFn = await listen<ProgressEvent>('progress', (event) => {
          setProgress(event.payload);
        });
        unlisten = unlistenFn;
      } catch {
        // Tauri event system not available (browser dev mode)
      }
    }

    setup();
    return () => { unlisten?.(); };
  }, []);

  const startOperation = () => { setRunning(true); setProgress(null); };
  const endOperation = () => { setRunning(false); setProgress(null); };

  return { progress, running, startOperation, endOperation };
}
