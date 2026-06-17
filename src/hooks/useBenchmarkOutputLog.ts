import { useEffect, useRef, useState, type Dispatch, type SetStateAction } from "react";
import { isTauri } from "@tauri-apps/api/core";
import type { ApiOutputEvent, BenchmarkOutputEvent, BenchmarkOutputLine } from "../types";

const MAX_OUTPUT_LINES = 1000;
const MAX_API_OUTPUT_CHARS = 300_000;

function formatLogTimestamp(date: Date): string {
  const hours = date.getHours().toString().padStart(2, "0");
  const minutes = date.getMinutes().toString().padStart(2, "0");
  const seconds = date.getSeconds().toString().padStart(2, "0");
  return `${hours}:${minutes}:${seconds}`;
}

function trimOutputLines(lines: BenchmarkOutputLine[]): BenchmarkOutputLine[] {
  return lines.length > MAX_OUTPUT_LINES ? lines.slice(-MAX_OUTPUT_LINES) : lines;
}

function trimApiMessage(message: string, header = ""): string {
  if (!header || !message.startsWith(header)) {
    return message.length > MAX_API_OUTPUT_CHARS ? message.slice(-MAX_API_OUTPUT_CHARS) : message;
  }

  const body = message.slice(header.length);
  return body.length > MAX_API_OUTPUT_CHARS
    ? `${header}${body.slice(-MAX_API_OUTPUT_CHARS)}`
    : message;
}

export function useBenchmarkOutputLog() {
  const nextId = useRef(1);
  const [outputLines, setOutputLines] = useState<BenchmarkOutputLine[]>([]);
  const [apiOutputLines, setApiOutputLines] = useState<BenchmarkOutputLine[]>([]);

  const newOutputLine = (message: string): BenchmarkOutputLine => ({
    id: nextId.current++,
    timestamp: formatLogTimestamp(new Date()),
    message,
  });

  const appendOutput = (
    setLines: Dispatch<SetStateAction<BenchmarkOutputLine[]>>,
    message: string,
    capMessage?: (message: string) => string,
  ) => {
    const normalized = message.trimEnd();
    if (!normalized) return;

    setLines((current) => trimOutputLines([...current, newOutputLine(capMessage?.(normalized) ?? normalized)]));
  };

  const appendApiOutput = (event: ApiOutputEvent) => {
    if (event.mode !== "append") {
      appendOutput(setApiOutputLines, event.message, trimApiMessage);
      return;
    }

    const header = event.header ?? "";
    setApiOutputLines((current) => {
      const targetIndex = current.length - 1;
      const target = current[targetIndex];
      if (!header || !target?.message.startsWith(header)) {
        return trimOutputLines([
          ...current,
          newOutputLine(trimApiMessage(`${header}\n${event.message}`, header)),
        ]);
      }

      const next = [...current];
      next[targetIndex] = {
        ...target,
        message: trimApiMessage(`${target.message}${event.message}`, header),
      };
      return trimOutputLines(next);
    });
  };

  useEffect(() => {
    let unlistenBenchmark: (() => void) | undefined;
    let unlistenApi: (() => void) | undefined;
    let cancelled = false;

    async function setup() {
      try {
        const { listen } = await import("@tauri-apps/api/event");
        const unlistenBenchmarkFn = await listen<BenchmarkOutputEvent>("benchmark-output", (event) => {
          appendOutput(setOutputLines, event.payload.message);
        });
        const unlistenApiFn = await listen<ApiOutputEvent>("api-output", (event) => {
          appendApiOutput(event.payload);
        });
        if (cancelled) {
          unlistenBenchmarkFn();
          unlistenApiFn();
          return;
        }
        unlistenBenchmark = unlistenBenchmarkFn;
        unlistenApi = unlistenApiFn;
      } catch {
        // Tauri event system is not available in browser-only runs.
      }
    }

    function handleBrowserOutput(event: Event) {
      const detail = (event as CustomEvent<BenchmarkOutputEvent>).detail;
      if (detail?.message) appendOutput(setOutputLines, detail.message);
    }

    function handleBrowserApiOutput(event: Event) {
      const detail = (event as CustomEvent<ApiOutputEvent>).detail;
      if (detail?.message) appendApiOutput(detail);
    }

    if (isTauri()) {
      setup();
    } else {
      window.addEventListener("benchmark-output", handleBrowserOutput);
      window.addEventListener("api-output", handleBrowserApiOutput);
    }
    return () => {
      cancelled = true;
      window.removeEventListener("benchmark-output", handleBrowserOutput);
      window.removeEventListener("api-output", handleBrowserApiOutput);
      unlistenBenchmark?.();
      unlistenApi?.();
    };
  }, []);

  return { outputLines, apiOutputLines };
}
