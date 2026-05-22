import { ModelPicker } from './ModelPicker';
import { QuantPresetMenu } from './QuantPresetMenu';
import { RecipeControls } from './RecipeControls';
import { ProgressBar } from './ProgressBar';
import { TestButton } from './TestButton';
import type { QuantType, ProgressEvent } from '../../types';

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
  onTest: () => void;
}

export function Toolbar({
  modelPath, hasModel, running, progress,
  onOpenModel, onSetAll, onSaveRecipe, onLoadRecipe, onExport, onTest,
}: ToolbarProps) {
  return (
    <>
      <ModelPicker modelPath={modelPath} onOpen={onOpenModel} disabled={running} />
      <QuantPresetMenu onSetAll={onSetAll} disabled={!hasModel || running} />
      <div className="flex-1" />
      <ProgressBar progress={progress} />
      <RecipeControls onSave={onSaveRecipe} onLoad={onLoadRecipe} onExport={onExport} disabled={!hasModel || running} />
      <TestButton onClick={onTest} disabled={!hasModel} running={running} />
    </>
  );
}
