import type { ProgressEvent, RecipeEvalPreset, RecipeTestMode } from "../../types";

interface RunControlsProps {
  hasModel: boolean;
  running: boolean;
  progress: ProgressEvent | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onTest: () => void;
}

export function RunControls({
  hasModel,
  running,
  progress,
  evalPreset,
  testMode,
  onEvalPresetChange,
  onTestModeChange,
  onTest,
}: RunControlsProps) {
  return (
    <div className="editor-run-controls">
      {progress && <span className="run-progress">{progress.message}</span>}
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
      <button
        type="button"
        className="editor-run-button"
        aria-label="Run recipe test"
        disabled={!hasModel || running}
        onClick={onTest}
        title={testMode === "compare_baseline" ? "Compare Recipe" : "Test Recipe"}
      >
        <span aria-hidden="true" />
      </button>
    </div>
  );
}
