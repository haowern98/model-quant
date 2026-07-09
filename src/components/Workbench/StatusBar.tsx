import type { BenchmarkRunId, ProgressEvent } from "../../types";
import type { EditorTab } from "./editorTabModel";

interface StatusBarProps {
  running: boolean;
  cancelling: boolean;
  statusMessage: string | null;
  progress: ProgressEvent | null;
  selectedRunIds: BenchmarkRunId[];
  activeEditor: EditorTab | null;
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
  const tensorEditor = props.activeEditor?.kind === "tensor-values" ? props.activeEditor : null;
  const valueCount = tensorEditor?.shape.reduce((total, dim) => total * dim, 1);

  return (
    <footer className="status-bar" aria-label="Status bar">
      {label ? (
        <div className="status-bar-item status-bar-run-status">
          <span className="codicon codicon-sync status-bar-sync" aria-hidden="true" />
          <span>{label}</span>
        </div>
      ) : null}
      {tensorEditor ? (
        <div className="status-bar-tensor-values">
          <div className="status-bar-item">Shape [{tensorEditor.shape.join(", ")}]</div>
          <div className="status-bar-item">Quant {tensorEditor.quant}</div>
          <div className="status-bar-item">Values {valueCount?.toLocaleString()}</div>
        </div>
      ) : null}
    </footer>
  );
}
