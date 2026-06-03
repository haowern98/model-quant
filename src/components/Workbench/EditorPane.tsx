import { TensorTable } from "../DetailPanel/TensorTable";
import type {
  ProgressEvent,
  QuantType,
  RecipeEvalPreset,
  RecipeProfile,
  RecipeTestMode,
  TensorInfo,
} from "../../types";
import { BottomPanel } from "./BottomPanel";
import { LayerTabs } from "./LayerTabs";
import { RunControls } from "./RunControls";

interface EditorPaneProps {
  modelPath: string | null;
  hasModel: boolean;
  running: boolean;
  progress: ProgressEvent | null;
  openLayers: number[];
  activeLayerIndex: number | null;
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  onSelectLayer: (layerIndex: number) => void;
  onCloseLayer: (layerIndex: number) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onTest: () => void;
}

function basename(path: string | null): string {
  if (!path) return "No GGUF opened";
  return path.split(/[\\/]/).pop() ?? path;
}

function layerTitle(layerIndex: number | null): string {
  if (layerIndex === null) return "No layer selected";
  if (layerIndex < 0) return "Global tensors";
  return `Layer ${layerIndex}`;
}

export function EditorPane({
  modelPath,
  hasModel,
  running,
  progress,
  openLayers,
  activeLayerIndex,
  tensors,
  assignments,
  profile,
  evalPreset,
  testMode,
  onSelectLayer,
  onCloseLayer,
  onAssignQuant,
  onEvalPresetChange,
  onTestModeChange,
  onTest,
}: EditorPaneProps) {
  const activeTitle = layerTitle(activeLayerIndex);
  const activeBreadcrumb =
    activeLayerIndex === null ? "workspace" : activeTitle;

  return (
    <main className="editor-pane">
      <div className="editor-tabs-bar">
        <LayerTabs
          openLayers={openLayers}
          activeLayerIndex={activeLayerIndex}
          onSelectLayer={onSelectLayer}
          onCloseLayer={onCloseLayer}
        />
        <RunControls
          hasModel={hasModel}
          running={running}
          progress={progress}
          evalPreset={evalPreset}
          testMode={testMode}
          onEvalPresetChange={onEvalPresetChange}
          onTestModeChange={onTestModeChange}
          onTest={onTest}
        />
      </div>

      <div className="editor-breadcrumbs">
        <span>{basename(modelPath)}</span>
        <span>&gt;</span>
        <span>{activeBreadcrumb}</span>
        <span>&gt;</span>
        <span>tensors</span>
      </div>

      <section className="tensor-editor-surface">
        <div className="tensor-editor-content">
          <div className="tensor-editor-title">
            <div>
              <h1>{activeTitle}</h1>
            </div>
          </div>
          <TensorTable
            tensors={tensors}
            assignments={assignments}
            onAssignQuant={onAssignQuant}
          />
        </div>
        <div className="editor-minimap" aria-hidden="true">
          {Array.from({ length: 24 }, (_, index) => (
            <span
              key={index}
              className={
                index % 5 === 0 ? "blue" : index % 7 === 0 ? "amber" : ""
              }
            />
          ))}
        </div>
      </section>

      <BottomPanel tensors={tensors} assignments={assignments} profile={profile} />
    </main>
  );
}
