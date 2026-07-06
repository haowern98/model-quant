import { useState, useEffect } from 'react';
import type { ProgressEvent } from '../types';

export function useProgress() {
  const [progress, setProgress] = useState<ProgressEvent | null>(null);
  const [running, setRunning] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

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

  const startOperation = (message: string | null = null) => {
    setRunning(true);
    setCancelling(false);
    setStatusMessage(message);
    setProgress(null);
  };
  const requestCancellation = () => {
    setCancelling(true);
    setStatusMessage("Cancelling test");
    setProgress({
      stage: "benchmarking",
      percent: 0,
      message: "Cancelling test...",
    });
  };
  const endOperation = () => {
    setRunning(false);
    setCancelling(false);
    setStatusMessage(null);
    setProgress(null);
  };

  return {
    progress,
    running,
    cancelling,
    statusMessage,
    startOperation,
    requestCancellation,
    endOperation,
  };
}
