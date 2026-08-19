import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type {
  BenchmarkOutputLine,
  QuantType,
  RecipeProfile,
  TensorInfo,
} from "../../types";
import { QUANT_TYPES, toTargetQuant } from "../../types";
import { estQuantSize, formatBytes } from "../../lib/format";
import { HardwarePanel } from "./HardwarePanel";
import type { ChatTraceSelection } from "./ChatTracePanel";

interface BottomPanelProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  outputLines: BenchmarkOutputLine[];
  apiOutputLines: BenchmarkOutputLine[];
  traceInspectorOpen: boolean;
  traceInspector: ChatTraceSelection | null;
  onClose: () => void;
  maximized: boolean;
  onToggleMaximized: () => void;
}

export function BottomPanel({
  tensors,
  assignments,
  profile,
  outputLines,
  apiOutputLines,
  traceInspectorOpen,
  traceInspector,
  onClose,
  maximized,
  onToggleMaximized,
}: BottomPanelProps) {
  const [activeTab, setActiveTab] =
    useState<"size" | "hardware" | "output" | "apiOutput" | "traceInspector">("hardware");
  useEffect(() => {
    if (!traceInspectorOpen && activeTab === "traceInspector") setActiveTab("hardware");
  }, [activeTab, traceInspectorOpen]);
  const totalTargetBytes = tensors.reduce((sum, tensor) => {
    const quant = assignments[tensor.name] ?? toTargetQuant(tensor.currentQuant);
    const bits = QUANT_TYPES.find((item) => item.value === quant)?.bitsPerWeight ?? 4.5;
    return sum + estQuantSize(tensor.shape, bits);
  }, 0);
  const f16Size = tensors.reduce((sum, tensor) => sum + estQuantSize(tensor.shape, 16), 0);
  const q8Size = tensors.reduce((sum, tensor) => sum + estQuantSize(tensor.shape, 8), 0);
  const q4Size = tensors.reduce((sum, tensor) => sum + estQuantSize(tensor.shape, 4.8), 0);

  return (
    <section className="bottom-panel" aria-label="Bottom panel">
      <div className="bottom-tabs" role="tablist">
        <button
          type="button"
          role="tab"
          className={activeTab === "hardware" ? "active" : ""}
          aria-label="HARDWARE"
          aria-selected={activeTab === "hardware"}
          onClick={() => setActiveTab("hardware")}
        >
          HARDWARE
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === "output" ? "active" : ""}
          aria-label="OUTPUT"
          aria-selected={activeTab === "output"}
          onClick={() => setActiveTab("output")}
        >
          OUTPUT
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === "apiOutput" ? "active" : ""}
          aria-label="API OUTPUT"
          aria-selected={activeTab === "apiOutput"}
          onClick={() => setActiveTab("apiOutput")}
        >
          API OUTPUT
        </button>
        {traceInspectorOpen ? (
          <button
            type="button"
            role="tab"
            className={activeTab === "traceInspector" ? "active" : ""}
            aria-label="TRACE INSPECTOR"
            aria-selected={activeTab === "traceInspector"}
            onClick={() => setActiveTab("traceInspector")}
          >
            TRACE INSPECTOR
          </button>
        ) : null}
        <button
          type="button"
          className="bottom-panel-action bottom-panel-fullscreen"
          aria-label={maximized ? "Restore bottom panel" : "Maximize bottom panel"}
          title={maximized ? "Restore bottom panel" : "Maximize bottom panel"}
          onClick={onToggleMaximized}
        >
          <span
            className={`codicon codicon-${maximized ? "screen-normal" : "screen-full"}`}
            aria-hidden="true"
          />
        </button>
        <button
          type="button"
          className="bottom-panel-action bottom-panel-close"
          aria-label="Hide bottom panel"
          title="Hide bottom panel"
          onClick={onClose}
        >
          <span className="tab-close codicon codicon-close" aria-hidden="true" />
        </button>
      </div>
      {activeTab === "hardware" ? (
        <HardwarePanel />
      ) : activeTab === "output" ? (
        <OutputPanel
          outputLines={outputLines}
          ariaLabel="Benchmark output"
          emptyMessage="No benchmark output yet."
        />
      ) : activeTab === "apiOutput" ? (
        <OutputPanel
          outputLines={apiOutputLines}
          ariaLabel="API output"
          emptyMessage="No API output yet."
        />
      ) : activeTab === "traceInspector" ? (
        <TraceInspector selection={traceInspector} />
      ) : (
        <div className="bottom-content">
          <Metric label="FP16" value={formatBytes(f16Size)} />
          <Metric label="Q8_0" value={formatBytes(q8Size)} />
          <Metric label="Recipe" value={formatBytes(totalTargetBytes)} accent />
          <Metric label="Q4_K_M" value={formatBytes(q4Size)} />
          <div className="bottom-note">
            {profile
              ? `Profiled VRAM estimate ${formatBytes(profile.vramEstimate * 1024 * 1024)}.`
              : "Ready. Quick/Default and Single/Compare are run configuration controls for the current recipe."}
          </div>
        </div>
      )}
    </section>
  );
}

function TraceInspector({ selection }: { selection: ChatTraceSelection | null }) {
  if (!selection) return <div className="trace-inspector-empty">Loading selected trace token…</div>;

  const generatedCandidates = selection.token.layers.map((layer) => ({
    layer: layer.layer,
    candidate: layer.candidates.find((candidate) => candidate.tokenId === selection.token.tokenId) ?? null,
  }));
  const values = generatedCandidates
    .map(({ candidate }) => candidate ? traceMetric(candidate, selection.view) : null)
    .filter((value): value is number => value !== null);
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const candidateLabel = selection.candidate ? displayTraceToken(selection.candidate.tokenText) : "No candidate";

  return (
    <div className="trace-inspector" aria-label="Trace Inspector">
      <div className="trace-inspector-title">Selected token: “{candidateLabel}” · Layer {selection.layer.layer}</div>
      <div className="trace-inspector-summary">
        <div><span>Generated output</span><strong>{displayTraceToken(selection.token.tokenText)}</strong><small>Token ID {selection.token.tokenId}</small></div>
        <div><span>Selected candidate</span><strong>{candidateLabel}</strong><small>{selection.candidate ? `Candidate ID ${selection.candidate.tokenId}` : "No table candidate selected"}</small></div>
        <div><span>At selected layer</span><strong>{selection.rank ? `Rank ${selection.rank}` : "Not captured"}</strong><small>{`${selection.view === "logit" ? "Logit" : "Probability"} ${selection.candidate ? formatTraceMetric(selection.candidate, selection.view) : "—"}`}</small></div>
      </div>
      <div className="trace-inspector-trajectory">
        <span>Generated-token trajectory · {selection.view === "logit" ? "Logit" : "Probability"}</span>
        <div className="trace-inspector-bars">
          {generatedCandidates.map(({ layer, candidate }) => {
            const value = candidate ? traceMetric(candidate, selection.view) : null;
            const height = value === null || maximum <= minimum ? 0.5 : (value - minimum) / (maximum - minimum);
            return (
              <div
                key={layer}
                className={`trace-inspector-bar${layer === selection.layer.layer ? " selected" : ""}${value === null ? " missing" : ""}`}
                style={{ height: `${Math.max(4, Math.round(height * 100))}%` }}
                title={value === null ? `Layer ${layer}: outside captured candidates` : `Layer ${layer}: ${formatTraceMetric(candidate!, selection.view)}`}
              />
            );
          })}
        </div>
      </div>
      <div className="trace-inspector-meta">{selection.token.layers.length} layers · {selection.layer.candidates.length} captured candidates/layer</div>
    </div>
  );
}

function displayTraceToken(token: string): string {
  return token.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
}

function traceMetric(candidate: NonNullable<ChatTraceSelection["candidate"]>, view: ChatTraceSelection["view"]): number {
  return view === "probability" ? candidate.probability ?? 0 : candidate.logit;
}

function formatTraceMetric(candidate: NonNullable<ChatTraceSelection["candidate"]>, view: ChatTraceSelection["view"]): string {
  const value = traceMetric(candidate, view);
  return view === "probability" && value > 0 && value < 0.0001 ? value.toExponential(2) : value.toFixed(view === "probability" ? 4 : 3);
}

function OutputPanel({
  outputLines,
  ariaLabel,
  emptyMessage,
}: {
  outputLines: BenchmarkOutputLine[];
  ariaLabel: string;
  emptyMessage: string;
}) {
  const outputRef = useRef<HTMLDivElement | null>(null);
  const followTailRef = useRef(true);

  useLayoutEffect(() => {
    const output = outputRef.current;
    if (!output || !followTailRef.current) return;
    output.scrollTop = output.scrollHeight;
  }, [outputLines]);

  return (
    <div
      ref={outputRef}
      className="bottom-output"
      role="log"
      aria-label={ariaLabel}
      onScroll={(event) => {
        const output = event.currentTarget;
        followTailRef.current =
          output.scrollTop + output.clientHeight >= output.scrollHeight - 24;
      }}
    >
      {outputLines.length === 0 ? (
        <p className="bottom-output-empty">{emptyMessage}</p>
      ) : (
        outputLines.map((line) => (
          <div className="bottom-output-line" key={line.id}>
            <span className="bottom-output-time">[{line.timestamp}]</span>
            <span>{line.message}</span>
          </div>
        ))
      )}
    </div>
  );
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`bottom-metric ${accent ? "accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
