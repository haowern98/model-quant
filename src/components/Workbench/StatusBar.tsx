import type { BenchmarkRunId, ProgressEvent } from "../../types";

interface StatusBarProps {
  running: boolean;
  cancelling: boolean;
  statusMessage: string | null;
  progress: ProgressEvent | null;
  selectedRunIds: BenchmarkRunId[];
}

function runStatusLabel({
  running,
  cancelling,
  statusMessage,
  progress,
  selectedRunIds,
}: StatusBarProps): string | null {
  if (cancelling) return statusMessage ?? "Cancelling";
  if (!running) return null;
  if (statusMessage) return statusMessage;

  const message = progress?.message.toLowerCase() ?? "";
  if (message.includes("gpqa")) return "GPQA running";
  if (message.includes("humaneval")) return "HumanEval running";
  if (selectedRunIds.includes("gpqa_diamond")) return "GPQA running";
  if (selectedRunIds.includes("humaneval")) return "HumanEval running";
  return progress?.message ?? "Benchmark running";
}

export function StatusBar(props: StatusBarProps) {
  const label = runStatusLabel(props);

  return (
    <footer className="status-bar" aria-label="Status bar">
      {label ? (
        <div className="status-bar-item status-bar-run-status">
          <span className="codicon codicon-sync status-bar-sync" aria-hidden="true" />
          <span>{label}</span>
        </div>
      ) : null}
    </footer>
  );
}
