import type {
  AssignPattern,
  ProgressEvent,
  QuantType,
  RecipeEvalPreset,
  RecipeProfile,
  RecipeTestMode,
  TensorInfo,
} from "../../types";
import { ActivityBar } from "./ActivityBar";
import { EditorPane } from "./EditorPane";
import { ExplorerPanel } from "./ExplorerPanel";

interface WorkbenchShellProps {
  modelPath: string | null;
  tensors: TensorInfo[];
  selectedTensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  activeLayerIndex: number | null;
  openLayers: number[];
  expandedLayers: Set<number>;
  running: boolean;
  progress: ProgressEvent | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  onOpenLayer: (layerIndex: number) => void;
  onToggleLayer: (layerIndex: number) => void;
  onCloseLayer: (layerIndex: number) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onAssignByPattern: (pattern: AssignPattern, quantType: QuantType) => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onTest: () => void;
  onSaveRecipe: () => void;
  onLoadRecipe: () => void;
  onExport: () => void;
}

export function WorkbenchShell({
  modelPath,
  tensors,
  selectedTensors,
  assignments,
  profile,
  activeLayerIndex,
  openLayers,
  expandedLayers,
  running,
  progress,
  evalPreset,
  testMode,
  onOpenLayer,
  onToggleLayer,
  onCloseLayer,
  onAssignQuant,
  onAssignByPattern,
  onEvalPresetChange,
  onTestModeChange,
  onTest,
  onSaveRecipe,
  onLoadRecipe,
  onExport,
}: WorkbenchShellProps) {
  return (
    <div className="workbench-shell">
      <ActivityBar />
      <ExplorerPanel
        modelPath={modelPath}
        tensors={tensors}
        activeLayerIndex={activeLayerIndex}
        expandedLayers={expandedLayers}
        running={running}
        onOpenLayer={onOpenLayer}
        onToggleLayer={onToggleLayer}
        onAssignByPattern={onAssignByPattern}
        onSaveRecipe={onSaveRecipe}
        onLoadRecipe={onLoadRecipe}
        onExport={onExport}
      />
      <EditorPane
        modelPath={modelPath}
        hasModel={tensors.length > 0}
        running={running}
        progress={progress}
        openLayers={openLayers}
        activeLayerIndex={activeLayerIndex}
        tensors={selectedTensors}
        assignments={assignments}
        profile={profile}
        evalPreset={evalPreset}
        testMode={testMode}
        onSelectLayer={onOpenLayer}
        onCloseLayer={onCloseLayer}
        onAssignQuant={onAssignQuant}
        onEvalPresetChange={onEvalPresetChange}
        onTestModeChange={onTestModeChange}
        onTest={onTest}
      />
    </div>
  );
}
