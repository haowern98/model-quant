import { useState } from "react";
import type { BenchmarkRunId, ProgressEvent } from "../../types";
import type { EditorTab } from "./editorTabModel";

interface StatusBarProps {
  running: boolean;
  cancelling: boolean;
  statusMessage: string | null;
  progress: ProgressEvent | null;
  selectedRunIds: BenchmarkRunId[];
  activeEditor: EditorTab | null;
  onTensorDecimalPlacesChange: (editorId: string, decimalPlaces: number) => void;
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
  const [precisionMenuEditorId, setPrecisionMenuEditorId] = useState<string | null>(null);
  const precisionMenuOpen = precisionMenuEditorId === tensorEditor?.id;

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
          <div
            className="status-bar-precision-control"
            onBlur={(event) => {
              if (event.relatedTarget instanceof Node && event.currentTarget.contains(event.relatedTarget)) return;
              setPrecisionMenuEditorId(null);
            }}
          >
            <button
              type="button"
              className="status-bar-item status-bar-precision-button"
              aria-expanded={precisionMenuOpen}
              aria-haspopup="listbox"
              onClick={() => setPrecisionMenuEditorId(precisionMenuOpen ? null : tensorEditor.id)}
            >
              {tensorEditor.decimalPlaces} dp
            </button>
            {precisionMenuOpen ? (
              <div className="status-bar-precision-menu" role="listbox" aria-label="Decimal places">
                {Array.from({ length: 9 }, (_, index) => index + 1).map((decimalPlaces) => (
                  <button
                    type="button"
                    key={decimalPlaces}
                    role="option"
                    aria-selected={decimalPlaces === tensorEditor.decimalPlaces}
                    onClick={() => {
                      props.onTensorDecimalPlacesChange(tensorEditor.id, decimalPlaces);
                      setPrecisionMenuEditorId(null);
                    }}
                  >
                    {decimalPlaces} dp
                  </button>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}
    </footer>
  );
}
