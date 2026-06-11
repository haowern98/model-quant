import { useEffect, useRef, useState } from "react";
import { isTauri } from "@tauri-apps/api/core";
import type { BenchmarkOutputEvent, BenchmarkOutputLine } from "../types";

function formatLogTimestamp(date: Date): string {
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

export function useBenchmarkOutputLog() {
  const nextId = useRef(1);
  const [outputLines, setOutputLines] = useState<BenchmarkOutputLine[]>([]);

  const appendOutput = (message: string) => {
    const normalized = message.trimEnd();
    if (!normalized) return;

    setOutputLines((current) => [
      ...current,
      {
        id: nextId.current++,
        timestamp: formatLogTimestamp(new Date()),
        message: normalized,
      },
    ]);
  };

  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    async function setup() {
      try {
        const { listen } = await import("@tauri-apps/api/event");
        const unlistenFn = await listen<BenchmarkOutputEvent>("benchmark-output", (event) => {
          appendOutput(event.payload.message);
        });
        if (cancelled) {
          unlistenFn();
          return;
        }
        unlisten = unlistenFn;
      } catch {
        // Tauri event system is not available in browser-only runs.
      }
    }

    function handleBrowserOutput(event: Event) {
      const detail = (event as CustomEvent<BenchmarkOutputEvent>).detail;
      if (detail?.message) appendOutput(detail.message);
    }

    if (isTauri()) {
      setup();
    } else {
      window.addEventListener("benchmark-output", handleBrowserOutput);
    }
    return () => {
      cancelled = true;
      window.removeEventListener("benchmark-output", handleBrowserOutput);
      unlisten?.();
    };
  }, []);

  return { outputLines };
}
