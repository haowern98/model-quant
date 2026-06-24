import { useEffect, useRef, useState } from "react";
import type {
  BenchmarkRunId,
  GpqaDiamondStatus,
  HumanEvalStatus,
  ProgressEvent,
  RecipeEvalPreset,
  RecipeTestMode,
} from "../../types";

interface RunControlsProps {
  hasModel: boolean;
  running: boolean;
  cancelling: boolean;
  progress: ProgressEvent | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  selectedRunIds: BenchmarkRunId[];
  gpqaStatus: GpqaDiamondStatus;
  humanevalStatus: HumanEvalStatus;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onToggleRunTarget: (target: BenchmarkRunId) => void;
  onNoTestsSelected: () => void;
  onTest: () => void;
  onCancelTest: () => void;
}

export function RunControls({
  hasModel,
  running,
  cancelling,
  progress,
  evalPreset,
  testMode,
  selectedRunIds,
  gpqaStatus,
  humanevalStatus,
  onEvalPresetChange,
  onTestModeChange,
  onToggleRunTarget,
  onNoTestsSelected,
  onTest,
  onCancelTest,
}: RunControlsProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;

    function handlePointerDown(event: PointerEvent) {
      if (!menuRef.current?.contains(event.target as Node)) {
        setMenuOpen(false);
      }
    }

    function handleKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") {
        setMenuOpen(false);
      }
    }

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  const runLabel = running ? "Cancel recipe test" : "Run recipe test";
  const runTitle = running
    ? cancelling
      ? "Cancelling test"
      : "Cancel test"
    : testMode === "compare_baseline"
      ? "Compare Recipe"
      : "Test Recipe";
  const hasSelectedRun = selectedRunIds.some(
    (id) => id === "ppl_check" || id === "gpqa_diamond" || id === "humaneval",
  );
  const hasSelectedApiBenchmark =
    selectedRunIds.includes("gpqa_diamond") || selectedRunIds.includes("humaneval");
  const runDisabled = cancelling || (!hasModel && hasSelectedRun && !hasSelectedApiBenchmark);
  const progressMessage =
    progress?.message.toLowerCase().includes("gpqa")
      ? "GPQA running"
      : progress?.message.toLowerCase().includes("humaneval")
        ? "HumanEval running"
        : progress?.message;

  return (
    <div className="editor-run-controls">
      {progressMessage && <span className="run-progress">{progressMessage}</span>}
      <select
        aria-label="Eval preset"
        value={evalPreset}
        disabled={!hasModel || running}
        onChange={(event) => onEvalPresetChange(event.target.value as RecipeEvalPreset)}
      >
        <option value="default">Default</option>
        <option value="quick">Quick</option>
      </select>
      <select
        aria-label="Test mode"
        value={testMode}
        disabled={!hasModel || running}
        onChange={(event) => onTestModeChange(event.target.value as RecipeTestMode)}
      >
        <option value="single">Single</option>
        <option value="compare_baseline">Compare</option>
      </select>
      <div
        ref={menuRef}
        className={`run-split-action ${menuOpen ? "open" : ""} ${running ? "running" : ""}`}
        role="group"
        aria-label="Recipe test controls"
      >
        <button
          type="button"
          className="run-split-primary"
          aria-label={runLabel}
          disabled={runDisabled}
          onClick={() => {
            setMenuOpen(false);
            if (running) {
              onCancelTest();
            } else if (!hasSelectedRun) {
              setMenuOpen(true);
              onNoTestsSelected();
            } else {
              onTest();
            }
          }}
          title={runTitle}
        >
          <span
            className={`codicon ${running ? "codicon-stop-circle" : "codicon-run-all"}`}
            aria-hidden="true"
          />
        </button>
        <button
          type="button"
          className="run-split-chevron"
          aria-label="Test run options"
          aria-expanded={menuOpen}
          aria-haspopup="menu"
          onClick={() => setMenuOpen((open) => !open)}
          title="Test run options"
        >
          <span className="codicon codicon-chevron-down" aria-hidden="true" />
        </button>
        {menuOpen && (
          <div
            className="run-action-menu"
            role="menu"
            aria-label="Test run options"
          >
            <div className="run-menu-section-label">LOCAL CHECKS</div>
            <RunMenuCheckbox
              label="PPL Check"
              status={hasModel ? "Ready" : "Open model"}
              checked={selectedRunIds.includes("ppl_check")}
              disabled={running}
              onClick={() => onToggleRunTarget("ppl_check")}
            />
            <div className="run-menu-separator" role="separator" />
            <div className="run-menu-section-label">OFFICIAL BENCHMARKS</div>
            <RunMenuCheckbox
              label="GPQA Diamond"
              status={gpqaStatus.statusLabel}
              checked={selectedRunIds.includes("gpqa_diamond")}
              disabled={running}
              onClick={() => onToggleRunTarget("gpqa_diamond")}
            />
            <RunMenuCheckbox
              label="HumanEval"
              status={humanevalStatus.statusLabel}
              checked={selectedRunIds.includes("humaneval")}
              disabled={running}
              onClick={() => onToggleRunTarget("humaneval")}
            />
            <RunMenuCheckbox label="MMLU-Pro" status="Download" disabled muted />
            <RunMenuCheckbox label="MMLU-Redux" status="Frozen" disabled muted />
            <RunMenuCheckbox label="SuperGPQA" status="Download" disabled muted />
            <RunMenuCheckbox
              label="Claw-Eval"
              status="Needs harness"
              disabled
              muted
            />
          </div>
        )}
      </div>
    </div>
  );
}

interface RunMenuCheckboxProps {
  label: string;
  status: string;
  checked?: boolean;
  muted?: boolean;
  disabled?: boolean;
  onClick?: () => void;
}

function RunMenuCheckbox({
  label,
  status,
  checked = false,
  muted = false,
  disabled = false,
  onClick,
}: RunMenuCheckboxProps) {
  return (
    <button
      type="button"
      className={`run-menu-item ${muted ? "muted" : ""}`}
      role="menuitemcheckbox"
      aria-checked={checked}
      disabled={disabled}
      onClick={onClick}
    >
      <span className="run-menu-check" aria-hidden="true">
        {checked && <span className="codicon codicon-check" />}
      </span>
      <span className="run-menu-label">{label}</span>
      <span className="run-menu-status">{status}</span>
    </button>
  );
}
