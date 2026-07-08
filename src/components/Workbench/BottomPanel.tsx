import { useLayoutEffect, useRef, useState } from "react";
import type {
  BenchmarkOutputLine,
  QuantType,
  RecipeProfile,
  TensorInfo,
} from "../../types";
import { QUANT_TYPES, toTargetQuant } from "../../types";
import { estQuantSize, formatBytes } from "../../lib/format";
import { HardwarePanel } from "./HardwarePanel";

interface BottomPanelProps {
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  outputLines: BenchmarkOutputLine[];
  apiOutputLines: BenchmarkOutputLine[];
}

export function BottomPanel({
  tensors,
  assignments,
  profile,
  outputLines,
  apiOutputLines,
}: BottomPanelProps) {
  const [activeTab, setActiveTab] =
    useState<"size" | "hardware" | "output" | "apiOutput">("size");
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
          className={activeTab === "size" ? "active" : ""}
          aria-label="SIZE PROFILE"
          aria-selected={activeTab === "size"}
          onClick={() => setActiveTab("size")}
        >
          SIZE PROFILE
        </button>
        <button
          type="button"
          role="tab"
          className={activeTab === "hardware" ? "active" : ""}
          aria-label="HARDWARE"
          aria-selected={activeTab === "hardware"}
          onClick={() => setActiveTab("hardware")}
        >
          <span className="codicon codicon-pulse" aria-hidden="true" />
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
