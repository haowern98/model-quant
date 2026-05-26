import { ModelPicker } from "./ModelPicker";
import { QuantPresetMenu } from "./QuantPresetMenu";
import { RecipeControls } from "./RecipeControls";
import { ProgressBar } from "./ProgressBar";
import { TestButton } from "./TestButton";
import type { QuantType, ProgressEvent, RecipeTestMode } from "../../types";

interface ToolbarProps {
  modelPath: string | null;
  hasModel: boolean;
  running: boolean;
  progress: ProgressEvent | null;
  onOpenModel: () => void;
  onSetAll: (qt: QuantType) => void;
  onSaveRecipe: () => void;
  onLoadRecipe: () => void;
  onExport: () => void;
  testMode: RecipeTestMode;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onTest: () => void;
}

export function Toolbar({
  modelPath,
  hasModel,
  running,
  progress,
  onOpenModel,
  onSetAll,
  onSaveRecipe,
  onLoadRecipe,
  onExport,
  testMode,
  onTestModeChange,
  onTest,
}: ToolbarProps) {
  return (
    <>
      <ModelPicker
        modelPath={modelPath}
        onOpen={onOpenModel}
        disabled={running}
      />
      <QuantPresetMenu onSetAll={onSetAll} disabled={!hasModel || running} />
      <div className="flex-1" />
      <ProgressBar progress={progress} />
      <RecipeControls
        onSave={onSaveRecipe}
        onLoad={onLoadRecipe}
        onExport={onExport}
        disabled={!hasModel || running}
      />
      <select
        value={testMode}
        onChange={(event) =>
          onTestModeChange(event.target.value as RecipeTestMode)
        }
        disabled={!hasModel || running}
        className="h-8 bg-bg-surface border border-border-default rounded px-2 text-sm text-text-primary disabled:opacity-40"
        aria-label="Test recipe mode"
      >
        <option value="single">Single</option>
        <option value="compare_baseline">Compare</option>
      </select>
      <TestButton
        mode={testMode}
        onClick={onTest}
        disabled={!hasModel}
        running={running}
      />
    </>
  );
}
