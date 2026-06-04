import { useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import type {
  AssignPattern,
  BenchmarkResult,
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
import type { EditorTab } from "./editorTabModel";

const EXPLORER_DEFAULT_WIDTH = 365;
const EXPLORER_MIN_WIDTH = 150;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

interface WorkbenchShellProps {
  modelPath: string | null;
  tensors: TensorInfo[];
  selectedTensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  activeLayerIndex: number | null;
  openEditors: EditorTab[];
  activeEditorId: string | null;
  benchmarkResult: BenchmarkResult | null;
  expandedLayers: Set<number>;
  running: boolean;
  progress: ProgressEvent | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  onOpenLayer: (layerIndex: number) => void;
  onOpenModel: () => void;
  onToggleLayer: (layerIndex: number) => void;
  onSelectEditor: (editorId: string) => void;
  onCloseEditor: (editorId: string) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onAssignByPattern: (pattern: AssignPattern, quantType: QuantType) => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onTest: () => void;
  onSaveRecipe: () => void;
  onLoadRecipe: () => void;
  onExport: () => void;
  onDiscardResults: () => void;
}

export function WorkbenchShell({
  modelPath,
  tensors,
  selectedTensors,
  assignments,
  profile,
  activeLayerIndex,
  openEditors,
  activeEditorId,
  benchmarkResult,
  expandedLayers,
  running,
  progress,
  evalPreset,
  testMode,
  onOpenLayer,
  onOpenModel,
  onToggleLayer,
  onSelectEditor,
  onCloseEditor,
  onAssignQuant,
  onAssignByPattern,
  onEvalPresetChange,
  onTestModeChange,
  onTest,
  onSaveRecipe,
  onLoadRecipe,
  onExport,
  onDiscardResults,
}: WorkbenchShellProps) {
  const shellRef = useRef<HTMLDivElement>(null);
  const [explorerWidth, setExplorerWidth] = useState(EXPLORER_DEFAULT_WIDTH);

  const explorerMaxWidth = () => {
    const shellWidth = shellRef.current?.getBoundingClientRect().width ?? 1280;
    return Math.max(EXPLORER_MIN_WIDTH, Math.floor(shellWidth * 0.5));
  };

  const startExplorerResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = explorerWidth;
    document.body.classList.add("resizing-explorer");

    const handleMove = (moveEvent: PointerEvent) => {
      setExplorerWidth(
        clamp(startWidth + moveEvent.clientX - startX, EXPLORER_MIN_WIDTH, explorerMaxWidth()),
      );
    };
    const stopResize = () => {
      document.body.classList.remove("resizing-explorer");
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", stopResize);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", stopResize);
  };

  return (
    <div
      ref={shellRef}
      className="workbench-shell"
      style={{ "--explorer-width": `${explorerWidth}px` } as CSSProperties}
    >
      <ActivityBar />
      <ExplorerPanel
        modelPath={modelPath}
        tensors={tensors}
        activeLayerIndex={activeLayerIndex}
        expandedLayers={expandedLayers}
        running={running}
        onOpenLayer={onOpenLayer}
        onOpenModel={onOpenModel}
        onToggleLayer={onToggleLayer}
        onAssignByPattern={onAssignByPattern}
        onSaveRecipe={onSaveRecipe}
        onLoadRecipe={onLoadRecipe}
        onExport={onExport}
      />
      <div
        className="resize-handle explorer-resizer"
        role="separator"
        aria-label="Resize Explorer"
        aria-orientation="vertical"
        aria-valuemin={EXPLORER_MIN_WIDTH}
        aria-valuemax={explorerMaxWidth()}
        aria-valuenow={Math.round(explorerWidth)}
        tabIndex={0}
        onPointerDown={startExplorerResize}
        onKeyDown={(event) => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          const direction = event.key === "ArrowLeft" ? -1 : 1;
          setExplorerWidth((width) =>
            clamp(width + direction * 10, EXPLORER_MIN_WIDTH, explorerMaxWidth()),
          );
        }}
      />
      <EditorPane
        modelPath={modelPath}
        hasModel={tensors.length > 0}
        running={running}
        progress={progress}
        openEditors={openEditors}
        activeEditorId={activeEditorId}
        benchmarkResult={benchmarkResult}
        tensors={selectedTensors}
        assignments={assignments}
        profile={profile}
        evalPreset={evalPreset}
        testMode={testMode}
        onSelectEditor={onSelectEditor}
        onCloseEditor={onCloseEditor}
        onAssignQuant={onAssignQuant}
        onEvalPresetChange={onEvalPresetChange}
        onTestModeChange={onTestModeChange}
        onTest={onTest}
        onSaveRecipe={onSaveRecipe}
        onExport={onExport}
        onDiscardResults={onDiscardResults}
      />
    </div>
  );
}
