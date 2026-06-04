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
  cancelling: boolean;
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
  onCancelTest: () => void;
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
  cancelling,
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
  onCancelTest,
  onSaveRecipe,
  onLoadRecipe,
  onExport,
  onDiscardResults,
}: WorkbenchShellProps) {
  const shellRef = useRef<HTMLDivElement>(null);
  const [explorerWidth, setExplorerWidth] = useState(EXPLORER_DEFAULT_WIDTH);
  const lastExpandedExplorerWidth = useRef(EXPLORER_DEFAULT_WIDTH);
  const explorerVisible = explorerWidth > 0;

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
      const requestedWidth = startWidth + moveEvent.clientX - startX;
      if (requestedWidth <= 0 || (startWidth > 0 && requestedWidth < EXPLORER_MIN_WIDTH)) {
        if (startWidth > 0) {
          lastExpandedExplorerWidth.current = EXPLORER_MIN_WIDTH;
        }
        setExplorerWidth(0);
        return;
      }

      const nextWidth = clamp(requestedWidth, EXPLORER_MIN_WIDTH, explorerMaxWidth());
      lastExpandedExplorerWidth.current = nextWidth;
      setExplorerWidth(nextWidth);
    };
    const stopResize = () => {
      document.body.classList.remove("resizing-explorer");
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", stopResize);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", stopResize);
  };

  const toggleExplorer = () => {
    if (explorerVisible) {
      lastExpandedExplorerWidth.current = explorerWidth;
      setExplorerWidth(0);
      return;
    }

    const restoredWidth = clamp(
      lastExpandedExplorerWidth.current,
      EXPLORER_MIN_WIDTH,
      explorerMaxWidth(),
    );
    lastExpandedExplorerWidth.current = restoredWidth;
    setExplorerWidth(restoredWidth);
  };

  return (
    <div
      ref={shellRef}
      className={`workbench-shell ${explorerVisible ? "" : "explorer-collapsed"}`}
      style={{ "--explorer-width": `${explorerWidth}px` } as CSSProperties}
    >
      <ActivityBar explorerVisible={explorerVisible} onToggleExplorer={toggleExplorer} />
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
        aria-valuemin={0}
        aria-valuemax={explorerMaxWidth()}
        aria-valuenow={Math.round(explorerWidth)}
        tabIndex={0}
        onPointerDown={startExplorerResize}
        onKeyDown={(event) => {
          if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
          event.preventDefault();
          if (event.key === "ArrowLeft" && explorerWidth <= EXPLORER_MIN_WIDTH) {
            lastExpandedExplorerWidth.current = EXPLORER_MIN_WIDTH;
            setExplorerWidth(0);
            return;
          }
          if (event.key === "ArrowRight" && explorerWidth === 0) {
            lastExpandedExplorerWidth.current = EXPLORER_MIN_WIDTH;
            setExplorerWidth(EXPLORER_MIN_WIDTH);
            return;
          }

          const direction = event.key === "ArrowLeft" ? -1 : 1;
          const nextWidth = clamp(
            explorerWidth + direction * 10,
            EXPLORER_MIN_WIDTH,
            explorerMaxWidth(),
          );
          lastExpandedExplorerWidth.current = nextWidth;
          setExplorerWidth(nextWidth);
        }}
      />
      <EditorPane
        modelPath={modelPath}
        hasModel={tensors.length > 0}
        running={running}
        cancelling={cancelling}
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
        onCancelTest={onCancelTest}
        onSaveRecipe={onSaveRecipe}
        onExport={onExport}
        onDiscardResults={onDiscardResults}
      />
    </div>
  );
}
