import { useState } from "react";
import type {
  GpqaDiamondStatus,
  HumanEvalStatus,
  TerminalBenchStatus,
} from "../../types";
import { ExplorerSectionHeader } from "./ExplorerTree";

interface TestingPanelProps {
  running: boolean;
  gpqaStatus: GpqaDiamondStatus;
  humanevalStatus: HumanEvalStatus;
  terminalBenchStatus: TerminalBenchStatus;
  gpqaEditorActive: boolean;
  humanevalEditorActive: boolean;
  terminalBenchEditorActive: boolean;
  onRefreshAllBenchmarks: () => void;
  onOpenGpqaDetails: () => void;
  onOpenGpqaDataset: () => void;
  onOpenHumanEvalDetails: () => void;
  onOpenTerminalBenchDetails: () => void;
}

type TestingSectionId = "benchmarks" | "environment" | "latestRuns";

export function TestingPanel({
  running,
  gpqaStatus,
  humanevalStatus,
  terminalBenchStatus,
  gpqaEditorActive,
  humanevalEditorActive,
  terminalBenchEditorActive,
  onRefreshAllBenchmarks,
  onOpenGpqaDetails,
  onOpenHumanEvalDetails,
  onOpenTerminalBenchDetails,
}: TestingPanelProps) {
  const [sections, setSections] = useState<Record<TestingSectionId, boolean>>({
    benchmarks: true,
    environment: true,
    latestRuns: true,
  });

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

      <section className="testing-section testing-benchmarks-section">
        <ExplorerSectionHeader
          label="BENCHMARKS"
          expanded={sections.benchmarks}
          onClick={() => toggleSection("benchmarks")}
          action={
            <button
              type="button"
              className="tree-action-button benchmark-refresh-button"
              aria-label="Refresh All"
              title="Refresh All"
              disabled={running}
              onClick={onRefreshAllBenchmarks}
            >
              <span className="codicon codicon-refresh" aria-hidden="true" />
            </button>
          }
        />
        {sections.benchmarks && (
          <div className="explorer-section-body benchmark-card-list testing-panel-body">
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
            <BenchmarkCard
              title="HumanEval"
              description="Python code generation benchmark"
              meta="EvalScope · 164 samples · pass@1"
              status={humanevalStatus.statusLabel}
              icon="code"
              active={humanevalEditorActive}
              onClick={onOpenHumanEvalDetails}
            />
            <BenchmarkCard
              title="Terminal-Bench 2.1"
              description="Terminal task benchmark"
              meta="Harbor - terminal-bench-2-1"
              status={terminalBenchStatus.statusLabel}
              icon="code"
              active={terminalBenchEditorActive}
              onClick={onOpenTerminalBenchDetails}
            />
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
            <TestingDetailRow label="GPQA Diamond" value="63.1%" />
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
  icon = "beaker",
  onClick,
}: {
  title: string;
  description: string;
  meta?: string;
  status: string;
  active?: boolean;
  icon?: "beaker" | "code";
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      className={`benchmark-card ${active ? "active" : ""}`}
      aria-label={`${title} ${status}`}
      onClick={onClick}
    >
      <span className={`benchmark-card-icon codicon codicon-${icon}`} aria-hidden="true" />
      <span className="benchmark-card-copy">
        <strong>{title}</strong>
        <span>{description}</span>
        {meta ? <small>{meta}</small> : null}
      </span>
      <span className="benchmark-card-status">{status}</span>
    </button>
  );
}
