import { useEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";
import type {
  AssignPattern,
  BenchmarkRunId,
  BenchmarkOutputLine,
  GpqaBenchmarkConfigInput,
  GpqaDiamondStatus,
  GpqaShotMode,
  HumanEvalStatus,
  ProgressEvent,
  QuantType,
  RecipeEvalPreset,
  RecipeProfile,
  RecipeTestMode,
  TerminalBenchBenchmarkConfigInput,
  TerminalBenchDatasetStatus,
  TerminalBenchStatus,
  TensorInfo,
} from "../../types";
import { ActivityBar, type ActivityId } from "./ActivityBar";
import { EditorPane } from "./EditorPane";
import { ExplorerPanel } from "./ExplorerPanel";
import type { EditorTab } from "./editorTabModel";
import { StatusBar } from "./StatusBar";
import { TestingPanel } from "./TestingPanel";

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
  expandedLayers: Set<number>;
  running: boolean;
  cancelling: boolean;
  statusMessage: string | null;
  progress: ProgressEvent | null;
  outputLines: BenchmarkOutputLine[];
  apiOutputLines: BenchmarkOutputLine[];
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  selectedRunIds: BenchmarkRunId[];
  gpqaStatus: GpqaDiamondStatus;
  humanevalStatus: HumanEvalStatus;
  terminalBenchStatus: TerminalBenchStatus;
  terminalBenchDatasetStatus: TerminalBenchDatasetStatus;
  gpqaShotMode: GpqaShotMode;
  gpqaConfig: GpqaBenchmarkConfigInput;
  humanevalConfig: GpqaBenchmarkConfigInput;
  terminalBenchConfig: TerminalBenchBenchmarkConfigInput;
  modelExplorerFocusVersion: number;
  onOpenLayer: (layerIndex: number) => void;
  onOpenModel: () => void;
  onToggleLayer: (layerIndex: number) => void;
  onSelectEditor: (editorId: string) => void;
  onCloseEditor: (editorId: string) => void;
  onReorderEditor: (editorId: string, beforeEditorId: string | null) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onAssignByPattern: (pattern: AssignPattern, quantType: QuantType) => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onToggleRunTarget: (target: BenchmarkRunId) => void;
  onNoTestsSelected: () => void;
  onGpqaShotModeChange: (mode: GpqaShotMode) => void;
  onGpqaConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onHumanEvalConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onTerminalBenchConfigChange: (config: TerminalBenchBenchmarkConfigInput) => void;
  onInstallGpqaHarness: () => void;
  onDownloadGpqaDataset: () => void;
  onDeleteGpqaDataset: () => void;
  onDeleteGpqaHarness: () => void;
  onRefreshGpqaStatus: () => Promise<void>;
  onRefreshAllBenchmarks: () => void;
  onBeginBenchmarkSetup: (message?: string | null) => void;
  onEndBenchmarkSetup: () => void;
  onOpenGpqaDetails: () => void;
  onOpenGpqaDataset: () => void;
  onOpenHumanEvalDetails: () => void;
  onOpenTerminalBenchDetails: () => void;
  onInstallTerminalBenchHarness: () => void;
  onDownloadTerminalBenchDataset: () => void;
  onDeleteTerminalBenchDataset: () => void;
  onRefreshTerminalBenchStatus: () => void;
  onRunHumanEvalBenchmark: () => void;
  onRunTerminalBenchBenchmark: () => void;
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
  expandedLayers,
  running,
  cancelling,
  statusMessage,
  progress,
  outputLines,
  apiOutputLines,
  evalPreset,
  testMode,
  selectedRunIds,
  gpqaStatus,
  humanevalStatus,
  terminalBenchStatus,
  terminalBenchDatasetStatus,
  gpqaShotMode,
  gpqaConfig,
  humanevalConfig,
  terminalBenchConfig,
  modelExplorerFocusVersion,
  onOpenLayer,
  onOpenModel,
  onToggleLayer,
  onSelectEditor,
  onCloseEditor,
  onReorderEditor,
  onAssignQuant,
  onAssignByPattern,
  onEvalPresetChange,
  onTestModeChange,
  onToggleRunTarget,
  onNoTestsSelected,
  onGpqaShotModeChange,
  onGpqaConfigChange,
  onHumanEvalConfigChange,
  onTerminalBenchConfigChange,
  onInstallGpqaHarness,
  onDownloadGpqaDataset,
  onDeleteGpqaDataset,
  onDeleteGpqaHarness,
  onRefreshGpqaStatus,
  onRefreshAllBenchmarks,
  onBeginBenchmarkSetup,
  onEndBenchmarkSetup,
  onOpenGpqaDetails,
  onOpenGpqaDataset,
  onOpenHumanEvalDetails,
  onOpenTerminalBenchDetails,
  onInstallTerminalBenchHarness,
  onDownloadTerminalBenchDataset,
  onDeleteTerminalBenchDataset,
  onRefreshTerminalBenchStatus,
  onRunHumanEvalBenchmark,
  onRunTerminalBenchBenchmark,
  onTest,
  onCancelTest,
  onSaveRecipe,
  onLoadRecipe,
  onExport,
  onDiscardResults,
}: WorkbenchShellProps) {
  const shellRef = useRef<HTMLDivElement>(null);
  const [explorerWidth, setExplorerWidth] = useState(EXPLORER_DEFAULT_WIDTH);
  const [activeActivity, setActiveActivity] = useState<ActivityId>("gguf");
  const lastExpandedExplorerWidth = useRef(EXPLORER_DEFAULT_WIDTH);
  const sidePanelVisible = explorerWidth > 0;
  const activeEditor = openEditors.find((editor) => editor.id === activeEditorId) ?? null;
  const gpqaEditorActive = activeEditor?.kind === "gpqa-details" || activeEditor?.kind === "gpqa-dataset";
  const humanevalEditorActive = activeEditor?.kind === "humaneval-details";
  const terminalBenchEditorActive = activeEditor?.kind === "terminal-bench-details";

  useEffect(() => {
    if (modelExplorerFocusVersion === 0) return;
    setActiveActivity("gguf");
    setExplorerWidth((width) => {
      if (width > 0) return width;
      const restoredWidth = clamp(
        lastExpandedExplorerWidth.current,
        EXPLORER_MIN_WIDTH,
        explorerMaxWidth(),
      );
      lastExpandedExplorerWidth.current = restoredWidth;
      return restoredWidth;
    });
  }, [modelExplorerFocusVersion]);

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

  const toggleSidePanel = () => {
    if (sidePanelVisible) {
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

  const selectActivity = (activity: ActivityId) => {
    if (activity !== "gguf" && activity !== "testing") return;
    if (activity === activeActivity) {
      toggleSidePanel();
      return;
    }

    setActiveActivity(activity);
    if (!sidePanelVisible) {
      const restoredWidth = clamp(
        lastExpandedExplorerWidth.current,
        EXPLORER_MIN_WIDTH,
        explorerMaxWidth(),
      );
      lastExpandedExplorerWidth.current = restoredWidth;
      setExplorerWidth(restoredWidth);
    }
  };

  return (
    <div
      ref={shellRef}
      className={`workbench-shell ${sidePanelVisible ? "" : "explorer-collapsed"}`}
      style={{ "--explorer-width": `${explorerWidth}px` } as CSSProperties}
    >
      <ActivityBar
        activeActivity={activeActivity}
        panelVisible={sidePanelVisible}
        onSelectActivity={selectActivity}
      />
      {activeActivity === "testing" ? (
        <TestingPanel
          running={running}
          gpqaStatus={gpqaStatus}
          humanevalStatus={humanevalStatus}
          terminalBenchStatus={terminalBenchStatus}
          gpqaEditorActive={gpqaEditorActive}
          humanevalEditorActive={humanevalEditorActive}
          terminalBenchEditorActive={terminalBenchEditorActive}
          onRefreshAllBenchmarks={onRefreshAllBenchmarks}
          onOpenGpqaDetails={onOpenGpqaDetails}
          onOpenGpqaDataset={onOpenGpqaDataset}
          onOpenHumanEvalDetails={onOpenHumanEvalDetails}
          onOpenTerminalBenchDetails={onOpenTerminalBenchDetails}
        />
      ) : (
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
      )}
      <div
        className="resize-handle explorer-resizer"
        role="separator"
        aria-label={activeActivity === "testing" ? "Resize Testing" : "Resize Explorer"}
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
        outputLines={outputLines}
        apiOutputLines={apiOutputLines}
        openEditors={openEditors}
        activeEditorId={activeEditorId}
        tensors={selectedTensors}
        assignments={assignments}
        profile={profile}
        evalPreset={evalPreset}
        testMode={testMode}
        selectedRunIds={selectedRunIds}
        gpqaStatus={gpqaStatus}
        humanevalStatus={humanevalStatus}
        terminalBenchStatus={terminalBenchStatus}
        terminalBenchDatasetStatus={terminalBenchDatasetStatus}
        gpqaShotMode={gpqaShotMode}
        gpqaConfig={gpqaConfig}
        humanevalConfig={humanevalConfig}
        terminalBenchConfig={terminalBenchConfig}
        onInstallGpqaHarness={onInstallGpqaHarness}
        onDownloadGpqaDataset={onDownloadGpqaDataset}
        onDeleteGpqaDataset={onDeleteGpqaDataset}
        onDeleteGpqaHarness={onDeleteGpqaHarness}
        onRefreshGpqaStatus={onRefreshGpqaStatus}
        onBeginBenchmarkSetup={onBeginBenchmarkSetup}
        onEndBenchmarkSetup={onEndBenchmarkSetup}
        onSelectEditor={onSelectEditor}
        onCloseEditor={onCloseEditor}
        onReorderEditor={onReorderEditor}
        onAssignQuant={onAssignQuant}
        onEvalPresetChange={onEvalPresetChange}
        onTestModeChange={onTestModeChange}
        onToggleRunTarget={onToggleRunTarget}
        onNoTestsSelected={onNoTestsSelected}
        onGpqaShotModeChange={onGpqaShotModeChange}
        onGpqaConfigChange={onGpqaConfigChange}
        onHumanEvalConfigChange={onHumanEvalConfigChange}
        onTerminalBenchConfigChange={onTerminalBenchConfigChange}
        onInstallTerminalBenchHarness={onInstallTerminalBenchHarness}
        onDownloadTerminalBenchDataset={onDownloadTerminalBenchDataset}
        onDeleteTerminalBenchDataset={onDeleteTerminalBenchDataset}
        onRefreshTerminalBenchStatus={onRefreshTerminalBenchStatus}
        onRunHumanEvalBenchmark={onRunHumanEvalBenchmark}
        onRunTerminalBenchBenchmark={onRunTerminalBenchBenchmark}
        onTest={onTest}
        onCancelTest={onCancelTest}
        onSaveRecipe={onSaveRecipe}
        onExport={onExport}
        onDiscardResults={onDiscardResults}
      />
      <StatusBar
        running={running}
        cancelling={cancelling}
        statusMessage={statusMessage}
        progress={progress}
        selectedRunIds={selectedRunIds}
      />
    </div>
  );
}
