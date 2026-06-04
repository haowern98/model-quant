import { useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import { TensorTable } from "../DetailPanel/TensorTable";
import type {
  BenchmarkResult,
  ProgressEvent,
  QuantType,
  RecipeEvalPreset,
  RecipeProfile,
  RecipeTestMode,
  TensorInfo,
} from "../../types";
import { EvalResultsView } from "../EvalResults/EvalResultsView";
import { BottomPanel } from "./BottomPanel";
import { EditorTabs } from "./EditorTabs";
import { RunControls } from "./RunControls";
import { editorTabLabel, type EditorTab } from "./editorTabModel";

const BOTTOM_PANEL_DEFAULT_HEIGHT = 143;
const BOTTOM_PANEL_MIN_HEIGHT = 64;

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

interface EditorPaneProps {
  modelPath: string | null;
  hasModel: boolean;
  running: boolean;
  progress: ProgressEvent | null;
  openEditors: EditorTab[];
  activeEditorId: string | null;
  benchmarkResult: BenchmarkResult | null;
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  onSelectEditor: (editorId: string) => void;
  onCloseEditor: (editorId: string) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onTest: () => void;
  onSaveRecipe: () => void;
  onExport: () => void;
  onDiscardResults: () => void;
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
  openEditors,
  activeEditorId,
  benchmarkResult,
  tensors,
  assignments,
  profile,
  evalPreset,
  testMode,
  onSelectEditor,
  onCloseEditor,
  onAssignQuant,
  onEvalPresetChange,
  onTestModeChange,
  onTest,
  onSaveRecipe,
  onExport,
  onDiscardResults,
}: EditorPaneProps) {
  const editorRef = useRef<HTMLElement>(null);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(BOTTOM_PANEL_DEFAULT_HEIGHT);
  const activeEditor =
    openEditors.find((editor) => editor.id === activeEditorId) ?? null;
  const activeLayerIndex =
    activeEditor?.kind === "layer" ? activeEditor.layerIndex : null;
  const activeTitle =
    activeEditor?.kind === "eval-results"
      ? "Eval Results"
      : layerTitle(activeLayerIndex);
  const activeBreadcrumb = activeEditor ? editorTabLabel(activeEditor) : "workspace";
  const showingResults = activeEditor?.kind === "eval-results" && benchmarkResult;

  const bottomPanelMaxHeight = () => {
    const editorHeight = editorRef.current?.getBoundingClientRect().height ?? 800;
    return Math.max(BOTTOM_PANEL_MIN_HEIGHT, Math.floor(editorHeight * 0.7));
  };

  const startBottomPanelResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startY = event.clientY;
    const startHeight = bottomPanelHeight;
    document.body.classList.add("resizing-bottom-panel");

    const handleMove = (moveEvent: PointerEvent) => {
      setBottomPanelHeight(
        clamp(startHeight + startY - moveEvent.clientY, BOTTOM_PANEL_MIN_HEIGHT, bottomPanelMaxHeight()),
      );
    };
    const stopResize = () => {
      document.body.classList.remove("resizing-bottom-panel");
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", stopResize);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", stopResize);
  };

  return (
    <main
      ref={editorRef}
      className="editor-pane"
      style={{ "--bottom-panel-height": `${bottomPanelHeight}px` } as CSSProperties}
    >
      <div className="editor-tabs-bar">
        <EditorTabs
          openEditors={openEditors}
          activeEditorId={activeEditorId}
          onSelectEditor={onSelectEditor}
          onCloseEditor={onCloseEditor}
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
        <span>{showingResults ? "benchmark" : "tensors"}</span>
      </div>

      {showingResults ? (
        <EvalResultsView
          result={benchmarkResult}
          onSave={onSaveRecipe}
          onExport={onExport}
          onDiscard={onDiscardResults}
        />
      ) : (
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
      )}

      <div
        className="resize-handle bottom-panel-resizer"
        role="separator"
        aria-label="Resize bottom panel"
        aria-orientation="horizontal"
        aria-valuemin={BOTTOM_PANEL_MIN_HEIGHT}
        aria-valuemax={bottomPanelMaxHeight()}
        aria-valuenow={Math.round(bottomPanelHeight)}
        tabIndex={0}
        onPointerDown={startBottomPanelResize}
        onKeyDown={(event) => {
          if (event.key !== "ArrowUp" && event.key !== "ArrowDown") return;
          event.preventDefault();
          const direction = event.key === "ArrowUp" ? 1 : -1;
          setBottomPanelHeight((height) =>
            clamp(height + direction * 10, BOTTOM_PANEL_MIN_HEIGHT, bottomPanelMaxHeight()),
          );
        }}
      />
      <BottomPanel tensors={tensors} assignments={assignments} profile={profile} />
    </main>
  );
}
