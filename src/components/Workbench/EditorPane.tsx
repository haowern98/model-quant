import { useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
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
  const editorRef = useRef<HTMLElement>(null);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(BOTTOM_PANEL_DEFAULT_HEIGHT);
  const activeTitle = layerTitle(activeLayerIndex);
  const activeBreadcrumb =
    activeLayerIndex === null ? "workspace" : activeTitle;

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
