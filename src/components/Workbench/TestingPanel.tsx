import { useState } from "react";
import type {
  BenchmarkRunId,
  BenchmarkResult,
  GpqaDiamondStatus,
  ProgressEvent,
  QuantType,
  RecipeEvalPreset,
  RecipeTestMode,
} from "../../types";
import { ExplorerSectionHeader, ExplorerTreeRow } from "./ExplorerTree";

interface TestingPanelProps {
  modelPath: string | null;
  assignments: Record<string, QuantType>;
  benchmarkResult: BenchmarkResult | null;
  running: boolean;
  cancelling: boolean;
  progress: ProgressEvent | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  selectedRunIds: BenchmarkRunId[];
  gpqaStatus: GpqaDiamondStatus;
  onToggleRunTarget: (target: BenchmarkRunId) => void;
  onInstallGpqaHarness: () => void;
  onOpenGpqaDetails: () => void;
  onOpenGpqaDataset: () => void;
}

type TestingSectionId = "localChecks" | "benchmarks" | "environment" | "latestRuns";
type TestingBenchmarkId = "gpqaDiamond";

function modeLabel(mode: RecipeTestMode): string {
  return mode === "compare_baseline" ? "Compare" : "Single";
}

function statusLabel({
  running,
  cancelling,
  progress,
  benchmarkResult,
}: Pick<TestingPanelProps, "running" | "cancelling" | "progress" | "benchmarkResult">) {
  if (cancelling) return "Cancelling";
  if (running) return progress?.message ?? "Running";
  if (benchmarkResult) return "Latest run ready";
  return "Ready";
}

export function TestingPanel({
  modelPath,
  assignments,
  benchmarkResult,
  running,
  cancelling,
  progress,
  testMode,
  selectedRunIds,
  gpqaStatus,
  onToggleRunTarget,
  onOpenGpqaDetails,
  onOpenGpqaDataset,
}: TestingPanelProps) {
  const [sections, setSections] = useState<Record<TestingSectionId, boolean>>({
    localChecks: true,
    benchmarks: true,
    environment: true,
    latestRuns: true,
  });
  const [benchmarks, setBenchmarks] = useState<Record<TestingBenchmarkId, boolean>>({
    gpqaDiamond: true,
  });

  const changedTargetCount = Object.keys(assignments).length;
  const verifiedTargets =
    benchmarkResult?.requestedTargetCount !== undefined &&
    benchmarkResult?.verifiedTargetCount !== undefined
      ? `${benchmarkResult.verifiedTargetCount}/${benchmarkResult.requestedTargetCount}`
      : "0/0";

  const toggleSection = (section: TestingSectionId) => {
    setSections((current) => ({ ...current, [section]: !current[section] }));
  };

  const toggleBenchmark = (benchmark: TestingBenchmarkId) => {
    setBenchmarks((current) => ({ ...current, [benchmark]: !current[benchmark] }));
  };

  return (
    <aside className="testing-panel" aria-label="Testing">
      <div className="explorer-title">
        <span>MODEL EVALUATION</span>
        <button type="button" aria-label="Testing actions">
          ...
        </button>
      </div>

      <section className="testing-section">
        <ExplorerSectionHeader
          label="LOCAL CHECKS"
          expanded={sections.localChecks}
          onClick={() => toggleSection("localChecks")}
        />
        {sections.localChecks && (
          <div className="explorer-section-body">
            <ExplorerTreeRow
              label="PPL Check"
              right={modelPath ? "Ready" : "Open model"}
              expanded={false}
              active
              ariaLabel={`PPL Check ${modelPath ? "Ready" : "Open model"}`}
            />
            <button
              type="button"
              className="testing-detail-action"
              disabled={!modelPath || running}
              onClick={() => onToggleRunTarget("ppl_check")}
            >
              {selectedRunIds.includes("ppl_check") ? "Selected" : "Select"}
            </button>
            <TestingDetailRow label="Mode" value={modeLabel(testMode)} />
            <TestingDetailRow label="Changed targets" value={changedTargetCount} />
            <TestingDetailRow label="Verified targets" value={verifiedTargets} />
          </div>
        )}
      </section>

      <section className="testing-section">
        <ExplorerSectionHeader
          label="BENCHMARKS"
          expanded={sections.benchmarks}
          onClick={() => toggleSection("benchmarks")}
        />
        {sections.benchmarks && (
          <div className="explorer-section-body">
            <ExplorerTreeRow
              label="GPQA Diamond"
              right={gpqaStatus.statusLabel}
              expanded={benchmarks.gpqaDiamond}
              onToggle={() => toggleBenchmark("gpqaDiamond")}
              ariaLabel={`GPQA Diamond ${gpqaStatus.statusLabel}`}
            />
            {benchmarks.gpqaDiamond && (
              <>
                <TestingNavRow label="Details" ariaLabel="GPQA Diamond Details" onClick={onOpenGpqaDetails} />
                <TestingNavRow label="Dataset" ariaLabel="GPQA Diamond Dataset" onClick={onOpenGpqaDataset} />
              </>
            )}
            <ExplorerTreeRow label="MMLU-Pro" right="Download" ariaLabel="MMLU-Pro Download" />
            <ExplorerTreeRow label="MMLU-Redux" right="Frozen" ariaLabel="MMLU-Redux Frozen" />
            <ExplorerTreeRow label="SuperGPQA" right="Download" ariaLabel="SuperGPQA Download" />
            <ExplorerTreeRow label="Claw-Eval" right="Needs harness" ariaLabel="Claw-Eval Needs harness" />
          </div>
        )}
      </section>

      <section className="testing-section">
        <ExplorerSectionHeader
          label="ENVIRONMENT"
          expanded={sections.environment}
          onClick={() => toggleSection("environment")}
        />
        {sections.environment && (
          <div className="explorer-section-body">
            <TestingDetailRow label="Python" value={gpqaStatus.python ?? "Unavailable"} />
            <TestingDetailRow label="EvalScope" value={gpqaStatus.evalscope ?? "Unavailable"} />
            <TestingDetailRow label="Dataset cache" value="Open" />
          </div>
        )}
      </section>

      <section className="testing-section">
        <ExplorerSectionHeader
          label="LATEST RUNS"
          expanded={sections.latestRuns}
          onClick={() => toggleSection("latestRuns")}
        />
        {sections.latestRuns && (
          <div className="explorer-section-body">
            <ExplorerTreeRow label="GPQA Diamond" right="63.1%" ariaLabel="GPQA Diamond 63.1%" />
            <ExplorerTreeRow
              label="PPL Check"
              right={statusLabel({ running, cancelling, progress, benchmarkResult })}
              ariaLabel="PPL Check latest status"
            />
          </div>
        )}
      </section>
    </aside>
  );
}

function TestingDetailRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="tensor-child-row testing-detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function TestingNavRow({
  label,
  ariaLabel,
  onClick,
}: {
  label: string;
  ariaLabel: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="tensor-child-row testing-nav-row"
      aria-label={ariaLabel}
      onClick={onClick}
    >
      <span>{label}</span>
    </button>
  );
}
