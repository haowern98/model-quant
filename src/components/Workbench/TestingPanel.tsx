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
  gpqaEditorActive: boolean;
  onToggleRunTarget: (target: BenchmarkRunId) => void;
  onInstallGpqaHarness: () => void;
  onOpenGpqaDetails: () => void;
  onOpenGpqaDataset: () => void;
}

type TestingSectionId = "localChecks" | "benchmarks" | "environment" | "latestRuns";

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
  gpqaEditorActive,
  onToggleRunTarget,
  onOpenGpqaDetails,
}: TestingPanelProps) {
  const [sections, setSections] = useState<Record<TestingSectionId, boolean>>({
    localChecks: true,
    benchmarks: true,
    environment: true,
    latestRuns: true,
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
          <div className="explorer-section-body benchmark-card-list">
            <BenchmarkCard
              title="GPQA Diamond"
              description="Graduate-level science QA benchmark"
              meta="EvalScope · 198 samples · 5-shot CoT"
              status={gpqaStatus.statusLabel}
              active={gpqaEditorActive}
              onClick={onOpenGpqaDetails}
            />
            <BenchmarkCard title="MMLU-Pro" description="Multitask professional reasoning" status="Download" />
            <BenchmarkCard title="MMLU-Redux" description="Cleaned MMLU benchmark split" status="Frozen" />
            <BenchmarkCard title="SuperGPQA" description="Broad graduate-level QA benchmark" status="Download" />
            <BenchmarkCard title="Claw-Eval" description="Agentic tool-use evaluation" status="Needs harness" />
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

function BenchmarkCard({
  title,
  description,
  meta,
  status,
  active,
  onClick,
}: {
  title: string;
  description: string;
  meta?: string;
  status: string;
  active?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      className={`benchmark-card ${active ? "active" : ""}`}
      aria-label={`${title} ${status}`}
      onClick={onClick}
    >
      <span className="benchmark-card-icon codicon codicon-beaker" aria-hidden="true" />
      <span className="benchmark-card-copy">
        <strong>{title}</strong>
        <span>{description}</span>
        {meta ? <small>{meta}</small> : null}
      </span>
      <span className="benchmark-card-status">{status}</span>
    </button>
  );
}
