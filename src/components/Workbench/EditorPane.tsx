import {
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type CSSProperties,
  type PointerEvent as ReactPointerEvent,
  type ReactNode,
} from "react";
import { TensorTable } from "../DetailPanel/TensorTable";
import type {
  BenchmarkResult,
  BenchmarkOutputLine,
  BenchmarkRunId,
  GpqaBenchmarkConfigInput,
  GpqaDatasetRow,
  GpqaDiamondStatus,
  GpqaShotMode,
  GpqaThinkingMode,
  HumanEvalDatasetRow,
  HumanEvalDatasetStatus,
  HumanEvalStatus,
  MmmuProDatasetRow,
  MmmuProDatasetStatus,
  MmmuProStatus,
  ProgressEvent,
  QuantType,
  RecipeEvalPreset,
  RecipeProfile,
  RecipeTestMode,
  TerminalBenchBenchmarkConfigInput,
  TerminalBenchDatasetStatus,
  TerminalBenchDatasetRow,
  TerminalBenchStatus,
  TensorInfo,
  TensorValuesPreview,
} from "../../types";
import { EvalResultsView } from "../EvalResults/EvalResultsView";
import { BottomPanel } from "./BottomPanel";
import { EditorTabs } from "./EditorTabs";
import { RunControls } from "./RunControls";
import { editorTabLabel, type EditorTab } from "./editorTabModel";
import {
  deleteGpqaDiamondHarness,
  deleteHumanEvalDataset,
  deleteHumanEvalHarness,
  deleteMmmuProDataset,
  downloadHumanEvalDataset,
  downloadMmmuProDataset,
  getGpqaDiamondDatasetRows,
  getGpqaDiamondStatus,
  getHumanEvalDatasetRows,
  getMmmuProDatasetRows,
  getMmmuProDatasetStatus,
  getTerminalBenchDatasetRows,
  getHumanEvalDatasetStatus,
  getHumanEvalStatus,
  getTensorValues,
  installGpqaDiamondHarness,
  installHumanEvalHarness,
} from "../../lib/tauri-bridge";

const BOTTOM_PANEL_DEFAULT_HEIGHT = 143;
const BOTTOM_PANEL_MIN_HEIGHT = 64;
type GpqaBenchmarkTab = "details" | "dataset" | "configuration";
type HumanEvalBenchmarkTab = "details" | "dataset" | "configuration";
type TerminalBenchTab = "details" | "dataset" | "configuration";
type MmmuProTab = "details" | "dataset" | "configuration";

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

interface EditorPaneProps {
  modelPath: string | null;
  projectorPath: string | null;
  hasModel: boolean;
  running: boolean;
  cancelling: boolean;
  progress: ProgressEvent | null;
  outputLines: BenchmarkOutputLine[];
  apiOutputLines: BenchmarkOutputLine[];
  openEditors: EditorTab[];
  activeEditorId: string | null;
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  selectedRunIds: BenchmarkRunId[];
  gpqaStatus: GpqaDiamondStatus;
  humanevalStatus: HumanEvalStatus;
  mmmuProStatus: MmmuProStatus;
  terminalBenchStatus: TerminalBenchStatus;
  terminalBenchDatasetStatus: TerminalBenchDatasetStatus;
  gpqaShotMode: GpqaShotMode;
  gpqaConfig: GpqaBenchmarkConfigInput;
  humanevalConfig: GpqaBenchmarkConfigInput;
  terminalBenchConfig: TerminalBenchBenchmarkConfigInput;
  onSelectEditor: (editorId: string) => void;
  onCloseEditor: (editorId: string) => void;
  onReorderEditor: (editorId: string, beforeEditorId: string | null) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onInstallGpqaHarness: () => void;
  onDownloadGpqaDataset: () => void;
  onDeleteGpqaDataset: () => void;
  onDeleteGpqaHarness: () => void;
  onRefreshGpqaStatus: () => Promise<void>;
  onBeginBenchmarkSetup: (message?: string | null) => void;
  onEndBenchmarkSetup: () => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onToggleRunTarget: (target: BenchmarkRunId) => void;
  onNoTestsSelected: () => void;
  onGpqaShotModeChange: (mode: GpqaShotMode) => void;
  onGpqaConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onHumanEvalConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onTerminalBenchConfigChange: (config: TerminalBenchBenchmarkConfigInput) => void;
  onInstallTerminalBenchHarness: () => void;
  onDownloadTerminalBenchDataset: () => void;
  onDeleteTerminalBenchDataset: () => void;
  onRefreshTerminalBenchStatus: () => void;
  onRunHumanEvalBenchmark: () => void;
  onRunTerminalBenchBenchmark: () => void;
  onTest: () => void;
  onCancelTest: () => void;
  onSaveRecipe: () => void;
  onExport: () => void;
  onDiscardResults: () => void;
  bottomPanelVisible: boolean;
  onHideBottomPanel: () => void;
}

function basename(path: string | null): string {
  if (!path) return "No GGUF opened";
  return path.split(/[\\/]/).pop() ?? path;
}

function benchmarkResultLabel(result: BenchmarkResult): string | null {
  if (result.testMode === "official_gpqa_diamond") return "GPQA Diamond";
  if (result.testMode === "official_humaneval") return "HumanEval";
  if (result.testMode === "official_terminal_bench") return "Terminal-Bench 2.1";
  return null;
}

export function EditorPane({
  modelPath,
  projectorPath,
  hasModel,
  running,
  cancelling,
  progress,
  outputLines,
  apiOutputLines,
  openEditors,
  activeEditorId,
  tensors,
  assignments,
  profile,
  evalPreset,
  testMode,
  selectedRunIds,
  gpqaStatus,
  humanevalStatus,
  mmmuProStatus,
  terminalBenchStatus,
  terminalBenchDatasetStatus,
  gpqaShotMode,
  gpqaConfig,
  humanevalConfig,
  terminalBenchConfig,
  onSelectEditor,
  onCloseEditor,
  onReorderEditor,
  onAssignQuant,
  onInstallGpqaHarness,
  onDownloadGpqaDataset,
  onDeleteGpqaDataset,
  onDeleteGpqaHarness,
  onRefreshGpqaStatus,
  onBeginBenchmarkSetup,
  onEndBenchmarkSetup,
  onEvalPresetChange,
  onTestModeChange,
  onToggleRunTarget,
  onNoTestsSelected,
  onGpqaShotModeChange,
  onGpqaConfigChange,
  onHumanEvalConfigChange,
  onTerminalBenchConfigChange,
  onInstallTerminalBenchHarness,
  onDownloadTerminalBenchDataset,
  onDeleteTerminalBenchDataset,
  onRefreshTerminalBenchStatus,
  onRunHumanEvalBenchmark,
  onRunTerminalBenchBenchmark,
  onTest,
  onCancelTest,
  onSaveRecipe,
  onExport,
  onDiscardResults,
  bottomPanelVisible,
  onHideBottomPanel,
}: EditorPaneProps) {
  const editorRef = useRef<HTMLElement>(null);
  const [bottomPanelHeight, setBottomPanelHeight] = useState(BOTTOM_PANEL_DEFAULT_HEIGHT);
  const [bottomPanelMaximized, setBottomPanelMaximized] = useState(false);
  const activeEditor =
    openEditors.find((editor) => editor.id === activeEditorId) ?? null;
  const activeTitle = activeEditor ? editorTabLabel(activeEditor) : "No layer selected";
  const activeBreadcrumb = activeEditor ? editorTabLabel(activeEditor) : "workspace";
  const activeResult = activeEditor?.kind === "eval-results" ? activeEditor.result : null;
  const showingGpqaDetails = activeEditor?.kind === "gpqa-details";
  const showingGpqaDataset = activeEditor?.kind === "gpqa-dataset";
  const showingGpqaBenchmark = showingGpqaDetails || showingGpqaDataset;
  const showingHumanEvalBenchmark = activeEditor?.kind === "humaneval-details";
  const showingTerminalBenchBenchmark = activeEditor?.kind === "terminal-bench-details";
  const showingMmmuProBenchmark = activeEditor?.kind === "mmmu-pro-details";
  const showingTensorValues = activeEditor?.kind === "tensor-values";
  const tensorValuesEditor = showingTensorValues
    ? (activeEditor as Extract<EditorTab, { kind: "tensor-values" }>)
    : null;
  const showingBenchmark =
    showingGpqaBenchmark || showingHumanEvalBenchmark || showingTerminalBenchBenchmark || showingMmmuProBenchmark;
  const activeResultBenchmark = activeResult ? benchmarkResultLabel(activeResult) : null;

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
      className={`editor-pane ${bottomPanelVisible ? "" : "bottom-panel-hidden"} ${
        bottomPanelVisible && bottomPanelMaximized ? "bottom-panel-maximized" : ""
      }`}
      style={{ "--bottom-panel-height": `${bottomPanelHeight}px` } as CSSProperties}
    >
      <div className="editor-tabs-bar">
        <EditorTabs
          openEditors={openEditors}
          activeEditorId={activeEditorId}
          onSelectEditor={onSelectEditor}
          onCloseEditor={onCloseEditor}
          onReorderEditor={onReorderEditor}
        />
        <RunControls
          hasModel={hasModel}
          running={running}
          cancelling={cancelling}
          progress={progress}
          evalPreset={evalPreset}
          testMode={testMode}
          selectedRunIds={selectedRunIds}
          gpqaStatus={gpqaStatus}
          humanevalStatus={humanevalStatus}
          mmmuProStatus={mmmuProStatus}
          terminalBenchStatus={terminalBenchStatus}
          onEvalPresetChange={onEvalPresetChange}
          onTestModeChange={onTestModeChange}
          onToggleRunTarget={onToggleRunTarget}
          onNoTestsSelected={onNoTestsSelected}
          onTest={onTest}
          onCancelTest={onCancelTest}
        />
      </div>

      <div className="editor-breadcrumbs">
        {activeResultBenchmark ? (
          <>
            <span>Benchmarks</span>
            <span>&gt;</span>
            <span>{activeResultBenchmark}</span>
            <span>&gt;</span>
            <span>Eval Results</span>
          </>
        ) : tensorValuesEditor ? (
          <>
            <span>{basename(tensorValuesEditor.source === "mmproj" ? projectorPath : modelPath)}</span>
            <span>&gt;</span>
            <span>{tensorValuesEditor.layerLabel}</span>
            <span>&gt;</span>
            <span>{tensorValuesEditor.tensorName}</span>
          </>
        ) : (
          <>
            <span>{showingBenchmark ? "Benchmarks" : basename(modelPath)}</span>
            <span>&gt;</span>
            <span>{activeBreadcrumb}</span>
          </>
        )}
      </div>

      {activeResult ? (
        <EvalResultsView
          result={activeResult}
          onSave={onSaveRecipe}
          onExport={onExport}
          onDiscard={onDiscardResults}
        />
      ) : showingGpqaBenchmark ? (
        <GpqaBenchmarkView
          key={activeEditor?.kind}
          initialTab={showingGpqaDataset ? "dataset" : "details"}
          status={gpqaStatus}
          shotMode={gpqaShotMode}
          config={gpqaConfig}
          running={running}
          onInstallHarness={onInstallGpqaHarness}
          onDownloadDataset={onDownloadGpqaDataset}
          onDeleteDataset={onDeleteGpqaDataset}
          onDeleteHarness={onDeleteGpqaHarness}
          onRefreshStatus={onRefreshGpqaStatus}
          onBeginSetup={onBeginBenchmarkSetup}
          onEndSetup={onEndBenchmarkSetup}
          onShotModeChange={onGpqaShotModeChange}
          onConfigChange={onGpqaConfigChange}
          onRunBenchmark={onTest}
        />
      ) : showingHumanEvalBenchmark ? (
        <HumanEvalBenchmarkView
          status={humanevalStatus}
          config={humanevalConfig}
          running={running}
          onBeginSetup={onBeginBenchmarkSetup}
          onEndSetup={onEndBenchmarkSetup}
          onConfigChange={onHumanEvalConfigChange}
          onRunBenchmark={onRunHumanEvalBenchmark}
        />
      ) : showingTerminalBenchBenchmark ? (
        <TerminalBenchView
          status={terminalBenchStatus}
          datasetStatus={terminalBenchDatasetStatus}
          config={terminalBenchConfig}
          running={running}
          onInstallHarness={onInstallTerminalBenchHarness}
          onDownloadDataset={onDownloadTerminalBenchDataset}
          onDeleteDataset={onDeleteTerminalBenchDataset}
          onRefreshStatus={onRefreshTerminalBenchStatus}
          onConfigChange={onTerminalBenchConfigChange}
          onRunBenchmark={onRunTerminalBenchBenchmark}
        />
      ) : showingMmmuProBenchmark ? (
        <MmmuProBenchmarkView
          status={mmmuProStatus}
          running={running}
          onBeginSetup={onBeginBenchmarkSetup}
          onEndSetup={onEndBenchmarkSetup}
        />
      ) : showingTensorValues ? (
        <TensorValuesView editor={activeEditor as Extract<EditorTab, { kind: "tensor-values" }>} />
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
              readOnly={activeEditor?.kind === "layer" && activeEditor.source === "mmproj"}
            />
          </div>
        </section>
      )}

      {bottomPanelVisible ? (
        <>
          {bottomPanelMaximized ? null : (
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
          )}
          <BottomPanel
            tensors={tensors}
            assignments={assignments}
            profile={profile}
            outputLines={outputLines}
            apiOutputLines={apiOutputLines}
            onClose={onHideBottomPanel}
            maximized={bottomPanelMaximized}
            onToggleMaximized={() => setBottomPanelMaximized((maximized) => !maximized)}
          />
        </>
      ) : null}
    </main>
  );
}

const TENSOR_VALUES_CHUNK_ROWS = 32;
const TENSOR_VALUES_CHUNK_COLS = 16;
const TENSOR_VALUES_EDGE_PX = 32;
const TENSOR_VALUES_ROW_HEIGHT = 28;
const TENSOR_VALUES_COL_WIDTH = 96;
const TENSOR_VALUES_PLACEHOLDER_ROWS = 3;
const TENSOR_VALUES_PLACEHOLDER_COLS = 3;
const TENSOR_VALUES_MAX_ROW_CHUNKS = 5;
const TENSOR_VALUES_MAX_COL_CHUNKS = 5;
const TENSOR_VALUES_EVICT_DELAY_MS = 150;

type TensorValueChunk = {
  rowOffset: number;
  colOffset: number;
  preview: TensorValuesPreview;
};

function sortedOffsets(offsets: number[]): number[] {
  return Array.from(new Set(offsets)).sort((a, b) => a - b);
}

function chunkCols(chunks: TensorValueChunk[], rowOffset: number, colOffset: number): number {
  return chunks.find((chunk) => chunk.rowOffset === rowOffset && chunk.colOffset === colOffset)?.preview.cols ?? 0;
}

function offsetsAround(offsets: number[], activeIndex: number, maxCount: number): number[] {
  if (offsets.length <= maxCount) return offsets;
  const before = Math.floor(maxCount / 2);
  const start = Math.max(0, Math.min(activeIndex - before, offsets.length - maxCount));
  return offsets.slice(start, start + maxCount);
}

function TensorValuesView({ editor }: { editor: Extract<EditorTab, { kind: "tensor-values" }> }) {
  const isVectorTensor = editor.shape.length === 1;
  const scrollRef = useRef<HTMLDivElement>(null);
  const loadingChunks = useRef(new Set<string>());
  const evictionTimer = useRef<number | null>(null);
  const lastScrollTop = useRef(0);
  const lastScrollLeft = useRef(0);
  const [chunks, setChunks] = useState<TensorValueChunk[]>([]);
  const [loadingBefore, setLoadingBefore] = useState(false);
  const [loadingAfter, setLoadingAfter] = useState(false);
  const [loadingLeft, setLoadingLeft] = useState(false);
  const [loadingRight, setLoadingRight] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadChunk = (nextRowOffset: number, nextColOffset: number) => {
    const key = `${nextRowOffset}:${nextColOffset}`;
    if (loadingChunks.current.has(key)) return Promise.resolve();
    loadingChunks.current.add(key);
    setError(null);
    return getTensorValues({
      source: editor.source,
      tensorName: editor.tensorName,
      rowOffset: isVectorTensor ? 0 : nextRowOffset,
      colOffset: isVectorTensor ? nextRowOffset : nextColOffset,
      rowCount: isVectorTensor ? 1 : TENSOR_VALUES_CHUNK_ROWS,
      colCount: isVectorTensor ? TENSOR_VALUES_CHUNK_ROWS : TENSOR_VALUES_CHUNK_COLS,
    })
      .then((result) => {
        const preview = isVectorTensor
          ? {
              ...result,
              rows: result.cols,
              cols: result.cols > 0 ? 1 : 0,
              totalRows: result.totalCols,
              totalCols: result.totalCols > 0 ? 1 : 0,
            }
          : result;
        if (preview.rows === 0 || preview.cols === 0) return;
        setChunks((current) => {
          if (current.some((chunk) => chunk.rowOffset === nextRowOffset && chunk.colOffset === nextColOffset)) {
            return current;
          }
          return [...current, { rowOffset: nextRowOffset, colOffset: nextColOffset, preview }].sort(
            (a, b) => a.rowOffset - b.rowOffset || a.colOffset - b.colOffset,
          );
        });
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        loadingChunks.current.delete(key);
      });
  };

  const loadRowChunk = (nextRowOffset: number, position: "before" | "after") => {
    const colOffsets = sortedOffsets(chunks.map((chunk) => chunk.colOffset));
    const colsToLoad = colOffsets.length > 0 ? colOffsets : [0];
    if (position === "before") setLoadingBefore(true);
    if (position === "after") setLoadingAfter(true);
    Promise.all(colsToLoad.map((colOffset) => loadChunk(nextRowOffset, colOffset))).finally(() => {
      if (position === "before") {
        const addedRows = chunks[0] ? chunks[0].rowOffset - nextRowOffset : TENSOR_VALUES_CHUNK_ROWS;
        requestAnimationFrame(() => {
          if (scrollRef.current) scrollRef.current.scrollTop += addedRows * TENSOR_VALUES_ROW_HEIGHT;
        });
        setLoadingBefore(false);
      } else {
        setLoadingAfter(false);
      }
    });
  };

  const loadColChunk = (nextColOffset: number, position: "left" | "right") => {
    const rowOffsets = sortedOffsets(chunks.map((chunk) => chunk.rowOffset));
    const rowsToLoad = rowOffsets.length > 0 ? rowOffsets : [0];
    if (position === "left") setLoadingLeft(true);
    if (position === "right") setLoadingRight(true);
    Promise.all(rowsToLoad.map((rowOffset) => loadChunk(rowOffset, nextColOffset))).finally(() => {
      if (position === "left") {
        const addedCols = chunks[0] ? chunks[0].colOffset - nextColOffset : TENSOR_VALUES_CHUNK_COLS;
        requestAnimationFrame(() => {
          if (scrollRef.current) scrollRef.current.scrollLeft += addedCols * TENSOR_VALUES_COL_WIDTH;
        });
        setLoadingLeft(false);
      } else {
        setLoadingRight(false);
      }
    });
  };

  useEffect(() => {
    loadingChunks.current.clear();
    if (evictionTimer.current !== null) window.clearTimeout(evictionTimer.current);
    lastScrollTop.current = 0;
    lastScrollLeft.current = 0;
    setChunks([]);
    setError(null);
    loadChunk(0, 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editor.source, editor.tensorName]);

  useEffect(() => {
    if (chunks.length === 0) return;
    if (evictionTimer.current !== null) window.clearTimeout(evictionTimer.current);
    evictionTimer.current = window.setTimeout(() => {
      if (
        loadingChunks.current.size > 0 ||
        loadingBefore ||
        loadingAfter ||
        loadingLeft ||
        loadingRight ||
        !scrollRef.current
      ) {
        return;
      }

      const scroll = scrollRef.current;
      setChunks((current) => {
        const currentRowOffsets = sortedOffsets(current.map((chunk) => chunk.rowOffset));
        const currentColOffsets = sortedOffsets(current.map((chunk) => chunk.colOffset));
        if (
          currentRowOffsets.length <= TENSOR_VALUES_MAX_ROW_CHUNKS &&
          currentColOffsets.length <= TENSOR_VALUES_MAX_COL_CHUNKS
        ) {
          return current;
        }

        const rowChunkHeight = TENSOR_VALUES_ROW_HEIGHT * TENSOR_VALUES_CHUNK_ROWS;
        const colChunkWidth = TENSOR_VALUES_COL_WIDTH * TENSOR_VALUES_CHUNK_COLS;
        const activeRowIndex = Math.min(currentRowOffsets.length - 1, Math.max(0, Math.floor(scroll.scrollTop / rowChunkHeight)));
        const activeColIndex = Math.min(currentColOffsets.length - 1, Math.max(0, Math.floor(scroll.scrollLeft / colChunkWidth)));
        const keptRows = offsetsAround(currentRowOffsets, activeRowIndex, TENSOR_VALUES_MAX_ROW_CHUNKS);
        const keptCols = offsetsAround(currentColOffsets, activeColIndex, TENSOR_VALUES_MAX_COL_CHUNKS);
        const keptRowSet = new Set(keptRows);
        const keptColSet = new Set(keptCols);
        const next = current.filter((chunk) => keptRowSet.has(chunk.rowOffset) && keptColSet.has(chunk.colOffset));
        if (next.length === current.length) return current;

        const removedRowsBefore = keptRows[0] - currentRowOffsets[0];
        const removedColsBefore = keptCols[0] - currentColOffsets[0];
        requestAnimationFrame(() => {
          if (!scrollRef.current) return;
          if (removedRowsBefore > 0) {
            scrollRef.current.scrollTop = Math.max(0, scrollRef.current.scrollTop - removedRowsBefore * TENSOR_VALUES_ROW_HEIGHT);
          }
          if (removedColsBefore > 0) {
            scrollRef.current.scrollLeft = Math.max(0, scrollRef.current.scrollLeft - removedColsBefore * TENSOR_VALUES_COL_WIDTH);
          }
          lastScrollTop.current = scrollRef.current.scrollTop;
          lastScrollLeft.current = scrollRef.current.scrollLeft;
        });
        return next;
      });
    }, TENSOR_VALUES_EVICT_DELAY_MS);

    return () => {
      if (evictionTimer.current !== null) window.clearTimeout(evictionTimer.current);
    };
  }, [chunks, loadingAfter, loadingBefore, loadingLeft, loadingRight]);

  const rowOffsets = sortedOffsets(chunks.map((chunk) => chunk.rowOffset));
  const colOffsets = sortedOffsets(chunks.map((chunk) => chunk.colOffset));
  const firstChunk = chunks.find((chunk) => chunk.rowOffset === rowOffsets[0] && chunk.colOffset === colOffsets[0]) ?? null;
  const lastRowChunk =
    chunks.find((chunk) => chunk.rowOffset === rowOffsets[rowOffsets.length - 1] && chunk.colOffset === colOffsets[0]) ??
    null;
  const lastColChunk =
    chunks.find((chunk) => chunk.rowOffset === rowOffsets[0] && chunk.colOffset === colOffsets[colOffsets.length - 1]) ??
    null;
  const totalRows = firstChunk?.preview.totalRows ?? 0;
  const totalCols = firstChunk?.preview.totalCols ?? 0;
  const firstRowOffset = rowOffsets[0] ?? 0;
  const firstColOffset = colOffsets[0] ?? 0;
  const loadedRowEnd = lastRowChunk ? lastRowChunk.rowOffset + lastRowChunk.preview.rows : 0;
  const loadedColEnd = lastColChunk ? lastColChunk.colOffset + lastColChunk.preview.cols : 0;

  const handlePreviewScroll = () => {
    const scroll = scrollRef.current;
    if (!scroll || !firstChunk || !lastRowChunk || !lastColChunk) return;
    const verticalMoved = scroll.scrollTop !== lastScrollTop.current;
    const horizontalMoved = scroll.scrollLeft !== lastScrollLeft.current;
    lastScrollTop.current = scroll.scrollTop;
    lastScrollLeft.current = scroll.scrollLeft;

    if (verticalMoved) {
      if (scroll.scrollTop <= TENSOR_VALUES_EDGE_PX && firstRowOffset > 0) {
        loadRowChunk(Math.max(0, firstRowOffset - TENSOR_VALUES_CHUNK_ROWS), "before");
        return;
      }
      if (
        scroll.scrollTop + scroll.clientHeight >= scroll.scrollHeight - TENSOR_VALUES_EDGE_PX &&
        loadedRowEnd < totalRows
      ) {
        loadRowChunk(loadedRowEnd, "after");
        return;
      }
    }

    if (horizontalMoved) {
      if (scroll.scrollLeft <= TENSOR_VALUES_EDGE_PX && firstColOffset > 0) {
        loadColChunk(Math.max(0, firstColOffset - TENSOR_VALUES_CHUNK_COLS), "left");
        return;
      }
      if (
        scroll.scrollLeft + scroll.clientWidth >= scroll.scrollWidth - TENSOR_VALUES_EDGE_PX &&
        loadedColEnd < totalCols
      ) {
        loadColChunk(loadedColEnd, "right");
      }
    }
  };

  const placeholderRows = Array.from({ length: TENSOR_VALUES_PLACEHOLDER_ROWS }, (_, row) => row);
  const placeholderCols = Array.from({ length: TENSOR_VALUES_PLACEHOLDER_COLS }, (_, col) => col);
  const initialPlaceholderCols = isVectorTensor ? 1 : TENSOR_VALUES_CHUNK_COLS;

  return (
    <section className="tensor-values-surface">
      <div className="tensor-editor-title">
        <h1>{editor.tensorName}</h1>
      </div>
      {error ? <span className="tensor-values-error">{error}</span> : null}
      <div className="tensor-values-grid-scroll" ref={scrollRef} onScroll={handlePreviewScroll}>
        {chunks.length > 0 ? (
          <table className="tensor-values-grid">
            <thead>
              <tr>
                <th></th>
                {loadingLeft && placeholderCols.map((col) => <th key={`left-${col}`}></th>)}
                {colOffsets.flatMap((colOffset) => {
                  const chunk = chunks.find((item) => item.rowOffset === rowOffsets[0] && item.colOffset === colOffset);
                  return Array.from({ length: chunk?.preview.cols ?? 0 }, (_, col) => (
                    <th key={`${colOffset}-${col}`}>col {colOffset + col}</th>
                  ));
                })}
                {loadingRight && placeholderCols.map((col) => <th key={`right-${col}`}></th>)}
              </tr>
            </thead>
            <tbody>
              {loadingBefore &&
                placeholderRows.map((row) => (
                  <tr className="tensor-values-placeholder-row" key={`before-${row}`}>
                    <th></th>
                    {loadingLeft &&
                      placeholderCols.map((col) => (
                        <td key={`left-${col}`}>
                          <span />
                        </td>
                      ))}
                    {colOffsets.flatMap((colOffset) =>
                      Array.from({ length: chunkCols(chunks, rowOffsets[0], colOffset) }, (_, col) => (
                        <td key={`${colOffset}-${col}`}>
                          <span />
                        </td>
                      )),
                    )}
                    {loadingRight &&
                      placeholderCols.map((col) => (
                        <td key={`right-${col}`}>
                          <span />
                        </td>
                      ))}
                  </tr>
                ))}
              {rowOffsets.flatMap((rowOffset) => {
                const rowChunk = chunks.find((chunk) => chunk.rowOffset === rowOffset && chunk.colOffset === colOffsets[0]);
                return Array.from({ length: rowChunk?.preview.rows ?? 0 }, (_, row) => (
                  <tr key={`${rowOffset}-${row}`}>
                    <th>row {rowOffset + row}</th>
                    {loadingLeft &&
                      placeholderCols.map((col) => (
                        <td className="tensor-values-placeholder-row" key={`left-${col}`}>
                          <span />
                        </td>
                      ))}
                    {colOffsets.flatMap((colOffset) => {
                      const chunk = chunks.find((item) => item.rowOffset === rowOffset && item.colOffset === colOffset);
                      return Array.from({ length: chunk?.preview.cols ?? 0 }, (_, col) => (
                        <td key={`${colOffset}-${col}`}>
                          {chunk?.preview.values[row * chunk.preview.cols + col]?.toFixed(editor.decimalPlaces) ?? ""}
                        </td>
                      ));
                    })}
                    {loadingRight &&
                      placeholderCols.map((col) => (
                        <td className="tensor-values-placeholder-row" key={`right-${col}`}>
                          <span />
                        </td>
                      ))}
                  </tr>
                ));
              })}
              {loadingAfter &&
                placeholderRows.map((row) => (
                  <tr className="tensor-values-placeholder-row" key={`after-${row}`}>
                    <th></th>
                    {loadingLeft &&
                      placeholderCols.map((col) => (
                        <td key={`left-${col}`}>
                          <span />
                        </td>
                      ))}
                    {colOffsets.flatMap((colOffset) =>
                      Array.from({ length: chunkCols(chunks, rowOffsets[0], colOffset) }, (_, col) => (
                        <td key={`${colOffset}-${col}`}>
                          <span />
                        </td>
                      )),
                    )}
                    {loadingRight &&
                      placeholderCols.map((col) => (
                        <td key={`right-${col}`}>
                          <span />
                        </td>
                      ))}
                  </tr>
                ))}
            </tbody>
          </table>
        ) : (
          <table className="tensor-values-grid">
            <tbody>
              {placeholderRows.map((row) => (
                <tr className="tensor-values-placeholder-row" key={row}>
                  <th></th>
                  {Array.from({ length: initialPlaceholderCols }, (_, col) => (
                    <td key={col}>
                      <span />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </section>
  );
}

function GpqaBenchmarkView({
  initialTab,
  status,
  shotMode,
  config,
  running,
  onInstallHarness,
  onDownloadDataset,
  onDeleteDataset,
  onDeleteHarness,
  onRefreshStatus,
  onBeginSetup,
  onEndSetup,
  onShotModeChange,
  onConfigChange,
  onRunBenchmark,
}: {
  initialTab: GpqaBenchmarkTab;
  status: GpqaDiamondStatus;
  shotMode: GpqaShotMode;
  config: GpqaBenchmarkConfigInput;
  running: boolean;
  onInstallHarness: () => void;
  onDownloadDataset: () => void;
  onDeleteDataset: () => void;
  onDeleteHarness: () => void;
  onRefreshStatus: () => Promise<void>;
  onBeginSetup: (message?: string | null) => void;
  onEndSetup: () => void;
  onShotModeChange: (mode: GpqaShotMode) => void;
  onConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onRunBenchmark: () => void;
}) {
  const [activeTab, setActiveTab] = useState<GpqaBenchmarkTab>(initialTab);
  const [datasetRows, setDatasetRows] = useState<GpqaDatasetRow[]>([]);
  const [datasetRowsError, setDatasetRowsError] = useState<string | null>(null);
  const [loadingDatasetRows, setLoadingDatasetRows] = useState(false);
  const harnessReady = status.python && status.evalscope;

  useEffect(() => {
    if (activeTab !== "dataset" || !status.datasetReady) {
      setDatasetRows([]);
      setDatasetRowsError(null);
      setLoadingDatasetRows(false);
      return;
    }

    let cancelled = false;
    setLoadingDatasetRows(true);
    getGpqaDiamondDatasetRows()
      .then((rows) => {
        if (cancelled) return;
        setDatasetRows(rows);
        setDatasetRowsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetRows([]);
        setDatasetRowsError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setLoadingDatasetRows(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeTab, status.datasetReady]);

  const updateIntegerField =
    (field: "contextWindow" | "sampleLimit" | "topK") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*$/.test(value)) onConfigChange({ ...config, [field]: value });
    };
  const updateDecimalField =
    (field: "temperature" | "repeatPenalty" | "topP" | "minP") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*(?:\.\d*)?$/.test(value)) {
        onConfigChange({ ...config, [field]: value });
      }
    };
  const updateSignedDecimalField =
    (field: "presencePenalty") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^-?\d*(?:\.\d*)?$/.test(value)) {
        onConfigChange({ ...config, [field]: value });
      }
    };
  const updateThinking = (thinking: GpqaThinkingMode) => {
    onConfigChange({ ...config, thinking });
  };
  const handleDatasetAction = () => {
    if (!status.datasetReady) {
      onDownloadDataset();
      return;
    }
    onDeleteDataset();
  };
  const handleHarnessAction = () => {
    if (!harnessReady) {
      onInstallHarness();
      return;
    }
    onDeleteHarness();
  };
  const handleRefreshStatus = async (message: string) => {
    onBeginSetup(message);
    try {
      await onRefreshStatus();
    } finally {
      onEndSetup();
    }
  };

  return (
    <section className="benchmark-editor-surface">
      <div className="benchmark-page">
        <div className="benchmark-page-header">
          <div className="benchmark-page-hero">
            <div className="benchmark-page-title">
              <h1>GPQA Diamond</h1>
              <div className="benchmark-page-meta">
                <span>EvalScope</span>
                <span>|</span>
                <span>gpqa_diamond</span>
                <span>|</span>
                <span>198 samples</span>
              </div>
              <p>Official GPQA Diamond harness for comparing local GGUF model behavior through the in-process chat API.</p>
              <div className="benchmark-page-actions">
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running || (!status.datasetReady && !harnessReady)}
                  onClick={handleDatasetAction}
                >
                  {status.datasetReady ? "Delete dataset" : "Download dataset"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running}
                  onClick={() => void handleRefreshStatus("Verifying hash")}
                >
                  Verify hash
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running}
                  onClick={handleHarnessAction}
                >
                  {harnessReady ? "Delete harness" : "Install harness"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running}
                  onClick={() => void handleRefreshStatus("Refreshing status")}
                >
                  Refresh
                </button>
                <button
                  type="button"
                  className="benchmark-action-button primary"
                  disabled={running || !status.ready}
                  onClick={onRunBenchmark}
                >
                  Run Benchmark
                </button>
              </div>
            </div>
          </div>
          <div className="benchmark-page-tabs" role="tablist" aria-label="GPQA Diamond sections">
            {(["details", "dataset", "configuration"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                className={activeTab === tab ? "active" : ""}
                role="tab"
                aria-selected={activeTab === tab}
                onClick={() => setActiveTab(tab)}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="benchmark-page-body">
          <div className="benchmark-page-main">
            {activeTab === "details" ? (
              <div className="benchmark-copy">
                <h2>About This Harness</h2>
                <p>
                  GPQA Diamond evaluates graduate-level science reasoning through EvalScope using the
                  app&apos;s in-process OpenAI-compatible chat API. It is intended for repeatable local
                  checks against GGUF models without launching a separate llama-server process.
                </p>
                <h2>About The Dataset</h2>
                <p>
                  The dataset contains multiple-choice science questions with expert-written answers.
                  Each run asks the model to produce a clean final answer that the harness can score
                  against the expected choice.
                </p>
              </div>
            ) : activeTab === "dataset" ? (
              <div className="benchmark-copy">
                <h2>Dataset Preview</h2>
                {!status.datasetReady ? (
                  <p>Download and verify the dataset to preview rows.</p>
                ) : loadingDatasetRows ? (
                  <p>Loading dataset rows...</p>
                ) : datasetRowsError ? (
                  <p>{datasetRowsError}</p>
                ) : datasetRows.length === 0 ? (
                  <p>No dataset rows found.</p>
                ) : (
                  <div className="benchmark-dataset-table" role="table" aria-label="GPQA Diamond dataset rows">
                    <div className="benchmark-dataset-row header" role="row">
                      <span role="columnheader">#</span>
                      <span role="columnheader">Question</span>
                      <span role="columnheader">Choices</span>
                      <span role="columnheader">Answer</span>
                    </div>
                    {datasetRows.map((row) => (
                      <div className="benchmark-dataset-row" role="row" key={row.index}>
                        <span role="cell">{row.index}</span>
                        <span role="cell">{row.question || "Unavailable"}</span>
                        <span role="cell">{row.choices.join("\n") || "Unavailable"}</span>
                        <span role="cell">{row.answer ?? "Unavailable"}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="benchmark-copy">
                <BenchmarkInfoSection title="Configuration">
                  <BenchmarkSelectRow
                    label="Shots"
                    selectLabel="GPQA Diamond shots"
                    value={shotMode}
                    onChange={onShotModeChange}
                    options={[
                      { value: "five_shot_cot", label: "5-shot CoT" },
                      { value: "zero_shot", label: "0-shot CoT" },
                    ]}
                  />
                  <BenchmarkInfoRow label="Reasoning" value="CoT" />
                  <BenchmarkSelectRow
                    label="Thinking"
                    selectLabel="GPQA Diamond thinking"
                    value={config.thinking}
                    onChange={updateThinking}
                    options={[
                      { value: "off", label: "Off" },
                      { value: "on", label: "On" },
                    ]}
                  />
                  <BenchmarkInputRow
                    label="Temperature"
                    inputLabel="GPQA Diamond temperature"
                    value={config.temperature}
                    placeholder="0"
                    inputMode="decimal"
                    onChange={updateDecimalField("temperature")}
                  />
                  <BenchmarkInputRow
                    label="Top K Sampling"
                    inputLabel="GPQA Diamond top K sampling"
                    value={config.topK}
                    placeholder="40"
                    inputMode="numeric"
                    onChange={updateIntegerField("topK")}
                  />
                  <BenchmarkInputRow
                    label="Repeat Penalty"
                    inputLabel="GPQA Diamond repeat penalty"
                    value={config.repeatPenalty}
                    placeholder="1.1"
                    inputMode="decimal"
                    onChange={updateDecimalField("repeatPenalty")}
                  />
                  <BenchmarkInputRow
                    label="Presence Penalty"
                    inputLabel="GPQA Diamond presence penalty"
                    value={config.presencePenalty}
                    placeholder="0"
                    inputMode="decimal"
                    onChange={updateSignedDecimalField("presencePenalty")}
                  />
                  <BenchmarkInputRow
                    label="Top P Sampling"
                    inputLabel="GPQA Diamond top P sampling"
                    value={config.topP}
                    placeholder="0.95"
                    inputMode="decimal"
                    onChange={updateDecimalField("topP")}
                  />
                  <BenchmarkInputRow
                    label="Min P Sampling"
                    inputLabel="GPQA Diamond min P sampling"
                    value={config.minP}
                    placeholder="0.05"
                    inputMode="decimal"
                    onChange={updateDecimalField("minP")}
                  />
                  <BenchmarkInputRow
                    label="Context window"
                    inputLabel="GPQA Diamond context window"
                    value={config.contextWindow}
                    placeholder="20000"
                    inputMode="numeric"
                    onChange={updateIntegerField("contextWindow")}
                  />
                  <BenchmarkInfoRow label="Batch size" value="1" />
                  <BenchmarkInputRow
                    label="Samples"
                    inputLabel="GPQA Diamond samples"
                    value={config.sampleLimit}
                    placeholder="198"
                    inputMode="numeric"
                    onChange={updateIntegerField("sampleLimit")}
                  />
                </BenchmarkInfoSection>
              </div>
            )}
          </div>
          <aside className="benchmark-page-side">
            <p className="benchmark-readiness">{status.detail}</p>
            <BenchmarkInfoSection title="Harness">
              <BenchmarkInfoRow label="Framework" value="EvalScope" />
              <BenchmarkInfoRow label="Dataset" value="gpqa_diamond" />
              <BenchmarkInfoRow label="Metric" value="acc" />
              <BenchmarkInfoRow label="Status" value={harnessReady ? "Installed" : status.statusLabel} />
              <BenchmarkInfoRow label="Python" value={status.python ?? "Unavailable"} />
              <BenchmarkInfoRow label="EvalScope" value={status.evalscope ?? "Unavailable"} />
            </BenchmarkInfoSection>
            <BenchmarkInfoSection title="GPQA Diamond Dataset">
              <BenchmarkInfoRow label="Downloaded" value={status.datasetPath ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Verified" value={status.datasetReady ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Samples" value="198" />
              <BenchmarkInfoRow label="License" value="CC-BY-4.0" />
              <BenchmarkInfoRow label="Official asset" value={status.datasetUrl} />
              <BenchmarkInfoRow label="Cache path" value={status.datasetPath ?? "Not downloaded"} />
              <BenchmarkInfoRow label="SHA256" value={status.datasetHash ?? "Unavailable"} />
              <BenchmarkInfoRow label="Expected SHA256" value={status.expectedDatasetHash} />
            </BenchmarkInfoSection>
          </aside>
        </div>
      </div>
    </section>
  );
}

function HumanEvalBenchmarkView({
  status,
  config,
  running,
  onBeginSetup,
  onEndSetup,
  onConfigChange,
  onRunBenchmark,
}: {
  status: HumanEvalStatus;
  config: GpqaBenchmarkConfigInput;
  running: boolean;
  onBeginSetup: (message?: string | null) => void;
  onEndSetup: () => void;
  onConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onRunBenchmark: () => void;
}) {
  const [activeTab, setActiveTab] = useState<HumanEvalBenchmarkTab>("details");
  const [viewStatus, setViewStatus] = useState(status);
  const [datasetStatus, setDatasetStatus] = useState<HumanEvalDatasetStatus>({
    datasetReady: false,
    datasetStatusLabel: "Missing",
    datasetPath: null,
    datasetHash: null,
    datasetUrl: "opencompass/humaneval",
    expectedDatasetHash: "EvalScope dataset cache marker",
  });
  const [busy, setBusy] = useState(false);
  const [datasetRows, setDatasetRows] = useState<HumanEvalDatasetRow[]>([]);
  const [datasetRowsError, setDatasetRowsError] = useState<string | null>(null);
  const [loadingDatasetRows, setLoadingDatasetRows] = useState(false);
  const harnessInstalled =
    viewStatus.statusLabel !== "Needs harness" && Boolean(viewStatus.python && viewStatus.evalscope);
  const updateIntegerField =
    (field: "contextWindow" | "sampleLimit" | "topK") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*$/.test(value)) onConfigChange({ ...config, [field]: value });
    };
  const updateDecimalField =
    (field: "temperature" | "repeatPenalty" | "topP" | "minP") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*(?:\.\d*)?$/.test(value)) onConfigChange({ ...config, [field]: value });
    };
  const updatePresencePenalty = (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.currentTarget.value;
    if (/^-?\d*(?:\.\d*)?$/.test(value)) {
      onConfigChange({ ...config, presencePenalty: value });
    }
  };

  useEffect(() => {
    setViewStatus(status);
    getHumanEvalDatasetStatus()
      .then(setDatasetStatus)
      .catch((error: unknown) => {
        setViewStatus((current) => ({
          ...current,
          detail: error instanceof Error ? error.message : String(error),
        }));
      });
  }, [status]);

  useEffect(() => {
    if (activeTab !== "dataset" || !datasetStatus.datasetReady) {
      setDatasetRows([]);
      setDatasetRowsError(null);
      setLoadingDatasetRows(false);
      return;
    }

    let cancelled = false;
    setLoadingDatasetRows(true);
    getHumanEvalDatasetRows()
      .then((rows) => {
        if (cancelled) return;
        setDatasetRows(rows);
        setDatasetRowsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetRows([]);
        setDatasetRowsError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setLoadingDatasetRows(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeTab, datasetStatus.datasetReady]);

  const refreshStatus = async (message = "Refreshing status") => {
    setBusy(true);
    onBeginSetup(message);
    try {
      const [nextStatus, nextDatasetStatus] = await Promise.all([
        getHumanEvalStatus(),
        getHumanEvalDatasetStatus(),
      ]);
      setViewStatus(nextStatus);
      setDatasetStatus(nextDatasetStatus);
      window.dispatchEvent(new Event("modelinspector:benchmark-harness-changed"));
    } catch (error) {
      setViewStatus((current) => ({ ...current, detail: (error as Error).message }));
    } finally {
      setBusy(false);
      onEndSetup();
    }
  };

  const changeHarness = async () => {
    setBusy(true);
    onBeginSetup(harnessInstalled ? "Deleting harness" : "Installing harness");
    try {
      setViewStatus(
        harnessInstalled ? await deleteHumanEvalHarness() : await installHumanEvalHarness(),
      );
      window.dispatchEvent(new Event("modelinspector:benchmark-harness-changed"));
    } catch (error) {
      setViewStatus((current) => ({ ...current, detail: (error as Error).message }));
    } finally {
      setBusy(false);
      onEndSetup();
    }
  };

  const changeDataset = async () => {
    setBusy(true);
    onBeginSetup(datasetStatus.datasetReady ? "Deleting dataset" : "Downloading dataset");
    try {
      setDatasetStatus(
        datasetStatus.datasetReady
          ? await deleteHumanEvalDataset()
          : await downloadHumanEvalDataset(),
      );
      window.dispatchEvent(new Event("modelinspector:benchmark-harness-changed"));
    } catch (error) {
      setViewStatus((current) => ({ ...current, detail: (error as Error).message }));
    } finally {
      setBusy(false);
      onEndSetup();
    }
  };

  return (
    <section className="benchmark-editor-surface">
      <div className="benchmark-page">
        <div className="benchmark-page-header">
          <div className="benchmark-page-hero">
            <div className="benchmark-page-title">
              <h1>HumanEval</h1>
              <div className="benchmark-page-meta">
                <span>EvalScope</span>
                <span>|</span>
                <span>humaneval</span>
                <span>|</span>
                <span>164 samples</span>
              </div>
              <p>Official HumanEval harness for checking Python code generation through the in-process chat API.</p>
              <div className="benchmark-page-actions">
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running || (!datasetStatus.datasetReady && !harnessInstalled)}
                  onClick={changeDataset}
                >
                  {datasetStatus.datasetReady ? "Delete dataset" : "Download dataset"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running}
                  onClick={() => void refreshStatus("Verifying hash")}
                >
                  Verify hash
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running}
                  onClick={changeHarness}
                >
                  {harnessInstalled ? "Delete harness" : "Install harness"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running}
                  onClick={() => void refreshStatus("Refreshing status")}
                >
                  Refresh
                </button>
                <button
                  type="button"
                  className="benchmark-action-button primary"
                  disabled={busy || running || !viewStatus.ready || !datasetStatus.datasetReady}
                  onClick={onRunBenchmark}
                >
                  Run Benchmark
                </button>
              </div>
            </div>
          </div>
          <div className="benchmark-page-tabs" role="tablist" aria-label="HumanEval sections">
            {(["details", "dataset", "configuration"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                className={activeTab === tab ? "active" : ""}
                role="tab"
                aria-selected={activeTab === tab}
                onClick={() => setActiveTab(tab)}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="benchmark-page-body">
          <div className="benchmark-page-main">
            {activeTab === "details" ? (
              <div className="benchmark-copy">
                <h2>About This Harness</h2>
                <p>
                  HumanEval evaluates Python function synthesis through EvalScope using the
                  app&apos;s in-process OpenAI-compatible chat API. Docker is required so generated
                  code is executed in a sandbox.
                </p>
                <h2>About The Dataset</h2>
                <p>
                  The dataset contains Python programming tasks with hidden tests. Each run asks
                  the model to produce code that can pass the task&apos;s tests.
                </p>
              </div>
            ) : activeTab === "dataset" ? (
              <div className="benchmark-copy">
                <h2>Dataset Preview</h2>
                {!datasetStatus.datasetReady ? (
                  <p>Download and verify the dataset to preview rows.</p>
                ) : loadingDatasetRows ? (
                  <p>Loading dataset rows...</p>
                ) : datasetRowsError ? (
                  <p>{datasetRowsError}</p>
                ) : datasetRows.length === 0 ? (
                  <p>No dataset rows found.</p>
                ) : (
                  <div className="benchmark-dataset-table" role="table" aria-label="HumanEval dataset rows">
                    <div className="benchmark-dataset-row header" role="row">
                      <span role="columnheader">#</span>
                      <span role="columnheader">Task</span>
                      <span role="columnheader">Prompt</span>
                      <span role="columnheader">Canonical solution</span>
                    </div>
                    {datasetRows.map((row) => (
                      <div className="benchmark-dataset-row" role="row" key={row.index}>
                        <span role="cell">{row.index}</span>
                        <span role="cell">
                          {row.taskId || "Unavailable"}
                          {row.entryPoint ? `\n${row.entryPoint}` : ""}
                        </span>
                        <span role="cell">{row.prompt || "Unavailable"}</span>
                        <span role="cell">{row.canonicalSolution || "Unavailable"}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="benchmark-copy">
                <BenchmarkInfoSection title="Configuration">
                  <BenchmarkSelectRow
                    label="Thinking"
                    selectLabel="HumanEval thinking"
                    value={config.thinking}
                    onChange={(thinking) => onConfigChange({ ...config, thinking })}
                    options={[
                      { value: "off", label: "Off" },
                      { value: "on", label: "On" },
                    ]}
                  />
                  <BenchmarkInputRow
                    label="Temperature"
                    inputLabel="HumanEval temperature"
                    value={config.temperature}
                    placeholder="0"
                    inputMode="decimal"
                    onChange={updateDecimalField("temperature")}
                  />
                  <BenchmarkInputRow
                    label="Top K Sampling"
                    inputLabel="HumanEval top K sampling"
                    value={config.topK}
                    placeholder="40"
                    inputMode="numeric"
                    onChange={updateIntegerField("topK")}
                  />
                  <BenchmarkInputRow
                    label="Repeat Penalty"
                    inputLabel="HumanEval repeat penalty"
                    value={config.repeatPenalty}
                    placeholder="1.1"
                    inputMode="decimal"
                    onChange={updateDecimalField("repeatPenalty")}
                  />
                  <BenchmarkInputRow
                    label="Presence Penalty"
                    inputLabel="HumanEval presence penalty"
                    value={config.presencePenalty}
                    placeholder="0"
                    inputMode="decimal"
                    onChange={updatePresencePenalty}
                  />
                  <BenchmarkInputRow
                    label="Top P Sampling"
                    inputLabel="HumanEval top P sampling"
                    value={config.topP}
                    placeholder="0.95"
                    inputMode="decimal"
                    onChange={updateDecimalField("topP")}
                  />
                  <BenchmarkInputRow
                    label="Min P Sampling"
                    inputLabel="HumanEval min P sampling"
                    value={config.minP}
                    placeholder="0.05"
                    inputMode="decimal"
                    onChange={updateDecimalField("minP")}
                  />
                  <BenchmarkInputRow
                    label="Context window"
                    inputLabel="HumanEval context window"
                    value={config.contextWindow}
                    placeholder="20000"
                    inputMode="numeric"
                    onChange={updateIntegerField("contextWindow")}
                  />
                  <BenchmarkInfoRow label="Batch size" value="1" />
                  <BenchmarkInputRow
                    label="Samples"
                    inputLabel="HumanEval samples"
                    value={config.sampleLimit}
                    placeholder="164"
                    inputMode="numeric"
                    onChange={updateIntegerField("sampleLimit")}
                  />
                </BenchmarkInfoSection>
              </div>
            )}
          </div>
          <aside className="benchmark-page-side">
            <p className="benchmark-readiness">{viewStatus.detail}</p>
            <BenchmarkInfoSection title="Harness">
              <BenchmarkInfoRow label="Framework" value="EvalScope" />
              <BenchmarkInfoRow label="Dataset" value="humaneval" />
              <BenchmarkInfoRow label="Metric" value="pass@1" />
              <BenchmarkInfoRow label="Status" value={viewStatus.statusLabel} />
              <BenchmarkInfoRow label="Python" value={viewStatus.python ?? "Unavailable"} />
              <BenchmarkInfoRow label="EvalScope" value={viewStatus.evalscope ?? "Unavailable"} />
              <BenchmarkInfoRow label="Docker" value={viewStatus.dockerReady ? viewStatus.docker ?? "Ready" : "Unavailable"} />
            </BenchmarkInfoSection>
            <BenchmarkInfoSection title="HumanEval Dataset">
              <BenchmarkInfoRow label="Downloaded" value={datasetStatus.datasetPath ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Verified" value={datasetStatus.datasetReady ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Samples" value="164" />
              <BenchmarkInfoRow label="License" value="MIT" />
              <BenchmarkInfoRow label="Official asset" value={datasetStatus.datasetUrl} />
              <BenchmarkInfoRow label="Cache path" value={datasetStatus.datasetPath ?? "Not downloaded"} />
              <BenchmarkInfoRow label="SHA256" value={datasetStatus.datasetHash ?? "Unavailable"} />
              <BenchmarkInfoRow label="Expected SHA256" value={datasetStatus.expectedDatasetHash} />
            </BenchmarkInfoSection>
          </aside>
        </div>
      </div>
    </section>
  );
}

function TerminalBenchView({
  status,
  datasetStatus,
  config,
  running,
  onInstallHarness,
  onDownloadDataset,
  onDeleteDataset,
  onRefreshStatus,
  onConfigChange,
  onRunBenchmark,
}: {
  status: TerminalBenchStatus;
  datasetStatus: TerminalBenchDatasetStatus;
  config: TerminalBenchBenchmarkConfigInput;
  running: boolean;
  onInstallHarness: () => void;
  onDownloadDataset: () => void;
  onDeleteDataset: () => void;
  onRefreshStatus: () => void;
  onConfigChange: (config: TerminalBenchBenchmarkConfigInput) => void;
  onRunBenchmark: () => void;
}) {
  const [activeTab, setActiveTab] = useState<TerminalBenchTab>("details");
  const [datasetRows, setDatasetRows] = useState<TerminalBenchDatasetRow[]>([]);
  const [datasetRowsError, setDatasetRowsError] = useState<string | null>(null);
  const [loadingDatasetRows, setLoadingDatasetRows] = useState(false);
  const updateIntegerField =
    (field: "topK" | "contextWindow" | "samples" | "runsPerTask" | "maxTurns" | "timeoutMultiplier") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*$/.test(value)) onConfigChange({ ...config, [field]: value });
    };
  const updateDecimalField =
    (field: "temperature" | "repeatPenalty" | "topP" | "minP") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*(?:\.\d*)?$/.test(value)) onConfigChange({ ...config, [field]: value });
    };
  const updatePresencePenalty = (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.currentTarget.value;
    if (/^-?\d*(?:\.\d*)?$/.test(value)) {
      onConfigChange({ ...config, presencePenalty: value });
    }
  };
  const handleDatasetAction = () => {
    if (datasetStatus.datasetReady) {
      onDeleteDataset();
      return;
    }
    onDownloadDataset();
  };

  useEffect(() => {
    if (activeTab !== "dataset" || !datasetStatus.datasetReady) {
      setDatasetRows([]);
      setDatasetRowsError(null);
      setLoadingDatasetRows(false);
      return;
    }

    let cancelled = false;
    setLoadingDatasetRows(true);
    getTerminalBenchDatasetRows()
      .then((rows) => {
        if (cancelled) return;
        setDatasetRows(rows);
        setDatasetRowsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetRows([]);
        setDatasetRowsError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setLoadingDatasetRows(false);
      });

    return () => {
      cancelled = true;
    };
  }, [activeTab, datasetStatus.datasetPath, datasetStatus.datasetReady]);

  return (
    <section className="benchmark-editor-surface">
      <div className="benchmark-page">
        <div className="benchmark-page-header">
          <div className="benchmark-page-hero">
            <div className="benchmark-page-title">
              <h1>Terminal-Bench 2.1</h1>
              <div className="benchmark-page-meta">
                <span>Harbor</span>
                <span>|</span>
                <span>terminal-bench-2-1</span>
                <span>|</span>
                <span>terminal tasks</span>
              </div>
              <p>Official Terminal-Bench 2.1 evaluation shell for terminal task execution through Harbor.</p>
              <div className="benchmark-page-actions">
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running || (!datasetStatus.datasetReady && !status.harborReady)}
                  onClick={handleDatasetAction}
                >
                  {datasetStatus.datasetReady ? "Delete dataset" : "Download dataset"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running}
                  onClick={onRefreshStatus}
                >
                  Verify hash
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running || status.harborReady}
                  onClick={onInstallHarness}
                >
                  {status.harborReady ? "Delete harness" : "Install harness"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={running}
                  onClick={onRefreshStatus}
                >
                  Refresh
                </button>
                <button
                  type="button"
                  className="benchmark-action-button primary"
                  disabled={running || !status.ready || !datasetStatus.datasetReady}
                  onClick={onRunBenchmark}
                >
                  Run Benchmark
                </button>
              </div>
            </div>
          </div>
          <div className="benchmark-page-tabs" role="tablist" aria-label="Terminal-Bench sections">
            {(["details", "dataset", "configuration"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                className={activeTab === tab ? "active" : ""}
                role="tab"
                aria-selected={activeTab === tab}
                onClick={() => setActiveTab(tab)}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="benchmark-page-body">
          <div className="benchmark-page-main">
            {activeTab === "details" ? (
              <div className="benchmark-copy">
                <h2>About This Harness</h2>
                <p>
                  Terminal-Bench 2.1 evaluates terminal task solving through Harbor. The first
                  runnable version will use Harbor&apos;s Terminus agent against the app&apos;s
                  in-process OpenAI-compatible chat API.
                </p>
                <h2>About The Dataset</h2>
                <p>
                  The dataset contains terminal tasks that run inside isolated environments. Docker
                  and Harbor support are required before this benchmark can run.
                </p>
              </div>
            ) : activeTab === "dataset" ? (
              <div className="benchmark-copy">
                <h2>Dataset Preview</h2>
                {!datasetStatus.datasetReady ? (
                  <p>Download and verify the Terminal-Bench dataset to preview tasks.</p>
                ) : loadingDatasetRows ? (
                  <p>Loading dataset rows...</p>
                ) : datasetRowsError ? (
                  <p>{datasetRowsError}</p>
                ) : datasetRows.length === 0 ? (
                  <p>No dataset rows found.</p>
                ) : (
                  <div
                    className="benchmark-dataset-table terminal-bench-dataset-table"
                    role="table"
                    aria-label="Terminal-Bench dataset rows"
                  >
                    <div className="benchmark-dataset-row header" role="row">
                      <span role="columnheader">#</span>
                      <span role="columnheader">Task</span>
                      <span role="columnheader">Instruction</span>
                      <span role="columnheader">Path</span>
                    </div>
                    {datasetRows.map((row) => (
                      <div className="benchmark-dataset-row" role="row" key={row.path}>
                        <span role="cell">{row.index}</span>
                        <span role="cell">{row.taskId || "Unavailable"}</span>
                        <span role="cell">{row.instruction || "Unavailable"}</span>
                        <span role="cell">{row.path || "Unavailable"}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="benchmark-copy">
                <BenchmarkInfoSection title="Configuration">
                  <BenchmarkSelectRow
                    label="Thinking"
                    selectLabel="Terminal-Bench thinking"
                    value={config.thinking}
                    onChange={(thinking) => onConfigChange({ ...config, thinking })}
                    options={[
                      { value: "off", label: "Off" },
                      { value: "on", label: "On" },
                    ]}
                  />
                  <BenchmarkInputRow
                    label="Temperature"
                    inputLabel="Terminal-Bench temperature"
                    value={config.temperature}
                    placeholder="0"
                    inputMode="decimal"
                    onChange={updateDecimalField("temperature")}
                  />
                  <BenchmarkInputRow
                    label="Top K Sampling"
                    inputLabel="Terminal-Bench top K sampling"
                    value={config.topK}
                    placeholder="40"
                    inputMode="numeric"
                    onChange={updateIntegerField("topK")}
                  />
                  <BenchmarkInputRow
                    label="Repeat Penalty"
                    inputLabel="Terminal-Bench repeat penalty"
                    value={config.repeatPenalty}
                    placeholder="1.1"
                    inputMode="decimal"
                    onChange={updateDecimalField("repeatPenalty")}
                  />
                  <BenchmarkInputRow
                    label="Presence Penalty"
                    inputLabel="Terminal-Bench presence penalty"
                    value={config.presencePenalty}
                    placeholder="0"
                    inputMode="decimal"
                    onChange={updatePresencePenalty}
                  />
                  <BenchmarkInputRow
                    label="Top P Sampling"
                    inputLabel="Terminal-Bench top P sampling"
                    value={config.topP}
                    placeholder="0.95"
                    inputMode="decimal"
                    onChange={updateDecimalField("topP")}
                  />
                  <BenchmarkInputRow
                    label="Min P Sampling"
                    inputLabel="Terminal-Bench min P sampling"
                    value={config.minP}
                    placeholder="0.05"
                    inputMode="decimal"
                    onChange={updateDecimalField("minP")}
                  />
                  <BenchmarkInputRow
                    label="Context window"
                    inputLabel="Terminal-Bench context window"
                    value={config.contextWindow}
                    placeholder="20000"
                    inputMode="numeric"
                    onChange={updateIntegerField("contextWindow")}
                  />
                  <BenchmarkInputRow
                    label="Samples"
                    inputLabel="Terminal-Bench samples"
                    value={config.samples}
                    placeholder="All"
                    inputMode="numeric"
                    onChange={updateIntegerField("samples")}
                  />
                  <BenchmarkInputRow
                    label="Runs per task"
                    inputLabel="Terminal-Bench runs per task"
                    value={config.runsPerTask}
                    placeholder="1"
                    inputMode="numeric"
                    onChange={updateIntegerField("runsPerTask")}
                  />
                  <BenchmarkInputRow
                    label="Max turns"
                    inputLabel="Terminal-Bench max turns"
                    value={config.maxTurns}
                    placeholder="1"
                    inputMode="numeric"
                    onChange={updateIntegerField("maxTurns")}
                  />
                  <BenchmarkInputRow
                    label="Timeout multiplier"
                    inputLabel="Terminal-Bench timeout multiplier"
                    value={config.timeoutMultiplier}
                    placeholder="3"
                    inputMode="numeric"
                    onChange={updateIntegerField("timeoutMultiplier")}
                  />
                </BenchmarkInfoSection>
              </div>
            )}
          </div>
          <aside className="benchmark-page-side">
            <p className="benchmark-readiness">{status.detail}</p>
            <BenchmarkInfoSection title="Harness">
              <BenchmarkInfoRow label="Framework" value="Harbor" />
              <BenchmarkInfoRow label="Dataset" value="terminal-bench-2-1" />
              <BenchmarkInfoRow label="Metric" value="pass@1" />
              <BenchmarkInfoRow label="Status" value={status.statusLabel} />
              <BenchmarkInfoRow label="Agent" value="terminus-2" />
              <BenchmarkInfoRow label="Harbor" value={status.harborReady ? status.harbor ?? "Ready" : "Unavailable"} />
              <BenchmarkInfoRow label="Docker" value={status.dockerReady ? status.docker ?? "Ready" : "Unavailable"} />
            </BenchmarkInfoSection>
            <BenchmarkInfoSection title="Terminal-Bench Dataset">
              <BenchmarkInfoRow label="Downloaded" value={datasetStatus.datasetPath ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Verified" value={datasetStatus.datasetReady ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Official asset" value={datasetStatus.datasetUrl} />
              <BenchmarkInfoRow label="Cache path" value={datasetStatus.datasetPath ?? "Not downloaded"} />
              <BenchmarkInfoRow label="SHA256" value={datasetStatus.datasetHash ?? "Unavailable"} />
              <BenchmarkInfoRow label="Expected SHA256" value={datasetStatus.expectedDatasetHash} />
            </BenchmarkInfoSection>
          </aside>
        </div>
      </div>
    </section>
  );
}

function MmmuProBenchmarkView({
  status,
  running,
  onBeginSetup,
  onEndSetup,
}: {
  status: MmmuProStatus;
  running: boolean;
  onBeginSetup: (message?: string | null) => void;
  onEndSetup: () => void;
}) {
  const [activeTab, setActiveTab] = useState<MmmuProTab>("details");
  const [datasetStatus, setDatasetStatus] = useState<MmmuProDatasetStatus>({
    datasetReady: false,
    datasetStatusLabel: "Missing",
    datasetPath: null,
    datasetHash: null,
    datasetUrl: "AI-ModelScope/MMMU_Pro",
    expectedDatasetHash: "EvalScope dataset cache marker",
  });
  const [harnessStatus, setHarnessStatus] = useState<GpqaDiamondStatus | null>(null);
  const [busy, setBusy] = useState(false);
  const [detail, setDetail] = useState<string | null>(null);
  const [datasetRows, setDatasetRows] = useState<MmmuProDatasetRow[]>([]);
  const [datasetRowsError, setDatasetRowsError] = useState<string | null>(null);
  const [loadingDatasetRows, setLoadingDatasetRows] = useState(false);
  const harnessInstalled = Boolean(harnessStatus?.python && harnessStatus?.evalscope);

  useEffect(() => {
    let cancelled = false;
    getMmmuProDatasetStatus()
      .then((nextDatasetStatus) => {
        if (cancelled) return;
        setDatasetStatus(nextDatasetStatus);
      })
      .catch((error: unknown) => {
        if (!cancelled) setDetail(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [status]);

  useEffect(() => {
    let cancelled = false;
    getGpqaDiamondStatus()
      .then((nextHarnessStatus) => {
        if (!cancelled) setHarnessStatus(nextHarnessStatus);
      })
      .catch((error: unknown) => {
        if (!cancelled) setDetail(error instanceof Error ? error.message : String(error));
      });
    return () => {
      cancelled = true;
    };
  }, [status]);

  useEffect(() => {
    if (activeTab !== "dataset" || !datasetStatus.datasetReady) {
      setDatasetRows([]);
      setDatasetRowsError(null);
      setLoadingDatasetRows(false);
      return;
    }

    let cancelled = false;
    setLoadingDatasetRows(true);
    getMmmuProDatasetRows()
      .then((rows) => {
        if (cancelled) return;
        setDatasetRows(rows);
        setDatasetRowsError(null);
      })
      .catch((error: unknown) => {
        if (cancelled) return;
        setDatasetRows([]);
        setDatasetRowsError(error instanceof Error ? error.message : String(error));
      })
      .finally(() => {
        if (!cancelled) setLoadingDatasetRows(false);
      });
    return () => {
      cancelled = true;
    };
  }, [activeTab, datasetStatus.datasetReady]);

  const refreshStatus = async (message = "Refreshing status") => {
    setBusy(true);
    onBeginSetup(message);
    try {
      const [datasetResult, harnessResult] = await Promise.allSettled([
        getMmmuProDatasetStatus().then(setDatasetStatus),
        getGpqaDiamondStatus().then(setHarnessStatus),
      ]);
      const failure = [datasetResult, harnessResult].find(
        (result): result is PromiseRejectedResult => result.status === "rejected",
      );
      if (failure) throw failure.reason;
      setDetail(null);
      window.dispatchEvent(new Event("modelinspector:benchmark-harness-changed"));
    } catch (error) {
      setDetail((error as Error).message);
    } finally {
      setBusy(false);
      onEndSetup();
    }
  };

  const changeHarness = async () => {
    setBusy(true);
    onBeginSetup(harnessInstalled ? "Deleting harness" : "Installing harness");
    try {
      const nextHarnessStatus = harnessInstalled
        ? await deleteGpqaDiamondHarness()
        : await installGpqaDiamondHarness();
      setHarnessStatus(nextHarnessStatus);
      setDetail(null);
      window.dispatchEvent(new Event("modelinspector:benchmark-harness-changed"));
    } catch (error) {
      setDetail((error as Error).message);
    } finally {
      setBusy(false);
      onEndSetup();
    }
  };

  const changeDataset = async () => {
    setBusy(true);
    onBeginSetup(datasetStatus.datasetReady ? "Deleting dataset" : "Downloading dataset");
    try {
      setDatasetStatus(
        datasetStatus.datasetReady ? await deleteMmmuProDataset() : await downloadMmmuProDataset(),
      );
      setDetail(null);
      window.dispatchEvent(new Event("modelinspector:benchmark-harness-changed"));
    } catch (error) {
      setDetail((error as Error).message);
    } finally {
      setBusy(false);
      onEndSetup();
    }
  };

  return (
    <section className="benchmark-editor-surface">
      <div className="benchmark-page">
        <div className="benchmark-page-header">
          <div className="benchmark-page-hero">
            <div className="benchmark-page-title">
              <h1>MMMU-Pro</h1>
              <div className="benchmark-page-meta">
                <span>EvalScope</span>
                <span>|</span>
                <span>mmmu_pro</span>
                <span>|</span>
                <span>1,730 samples</span>
              </div>
              <p>Official MMMU-Pro harness for multimodal visual reasoning through the in-process chat API.</p>
              <div className="benchmark-page-actions">
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running || (!datasetStatus.datasetReady && !harnessInstalled)}
                  onClick={changeDataset}
                >
                  {datasetStatus.datasetReady ? "Delete dataset" : "Download dataset"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running}
                  onClick={() => void refreshStatus("Verifying hash")}
                >
                  Verify hash
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running}
                  onClick={changeHarness}
                >
                  {harnessInstalled ? "Delete harness" : "Install harness"}
                </button>
                <button
                  type="button"
                  className="benchmark-action-button secondary"
                  disabled={busy || running}
                  onClick={() => void refreshStatus("Refreshing status")}
                >
                  Refresh
                </button>
                <button type="button" className="benchmark-action-button primary" disabled>
                  Run Benchmark
                </button>
              </div>
            </div>
          </div>
          <div className="benchmark-page-tabs" role="tablist" aria-label="MMMU-Pro sections">
            {(["details", "dataset", "configuration"] as const).map((tab) => (
              <button
                key={tab}
                type="button"
                className={activeTab === tab ? "active" : ""}
                role="tab"
                aria-selected={activeTab === tab}
                onClick={() => setActiveTab(tab)}
              >
                {tab.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div className="benchmark-page-body">
          <div className="benchmark-page-main">
            {activeTab === "details" ? (
              <div className="benchmark-copy">
                <h2>About This Benchmark</h2>
                <p>
                  MMMU-Pro evaluates multimodal visual reasoning through EvalScope using the
                  app&apos;s in-process OpenAI-compatible chat API and a compatible MMPROJ.
                </p>
                <h2>About The Dataset</h2>
                <p>
                  The dataset contains 1,730 multiple-choice visual reasoning tasks across 30
                  academic subjects. Dataset images are available in the preview after download.
                </p>
                <h2>Availability</h2>
                <p>Dataset setup and preview are available. MMMU-Pro execution is not wired yet.</p>
              </div>
            ) : activeTab === "dataset" ? (
              <div className="benchmark-copy">
                <h2>Dataset Preview</h2>
                {!datasetStatus.datasetReady ? (
                  <p>Download and verify the dataset to preview samples and images.</p>
                ) : loadingDatasetRows ? (
                  <p>Loading dataset samples...</p>
                ) : datasetRowsError ? (
                  <p>{datasetRowsError}</p>
                ) : datasetRows.length === 0 ? (
                  <p>No dataset samples found.</p>
                ) : (
                  <div className="benchmark-dataset-table" role="table" aria-label="MMMU-Pro dataset samples">
                    <div className="benchmark-dataset-row header" role="row">
                      <span role="columnheader">#</span>
                      <span role="columnheader">Task</span>
                      <span role="columnheader">Question</span>
                      <span role="columnheader">Images</span>
                    </div>
                    {datasetRows.map((row) => (
                      <div className="benchmark-dataset-row" role="row" key={row.index}>
                        <span role="cell">{row.index}</span>
                        <span role="cell">
                          {row.taskId || "Unavailable"}
                          {row.subject ? `\n${row.subject}` : ""}
                        </span>
                        <span role="cell">
                          {row.question || "Unavailable"}
                          {row.choices.length > 0 ? `\n${row.choices.join("\n")}` : ""}
                        </span>
                        <span role="cell">
                          {row.imageUrls.length > 0 ? (
                            <span style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                              {row.imageUrls.map((imageUrl, index) => (
                                <img
                                  key={`${row.index}-${index}`}
                                  src={imageUrl}
                                  alt={`${row.taskId || "MMMU-Pro sample"} image ${index + 1}`}
                                  style={{ width: 112, maxHeight: 84, objectFit: "contain" }}
                                />
                              ))}
                            </span>
                          ) : (
                            "No image"
                          )}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="benchmark-copy">
                <h2>Configuration</h2>
                <p>MMMU-Pro run configuration is not wired yet.</p>
              </div>
            )}
          </div>
          <aside className="benchmark-page-side">
            <p className="benchmark-readiness">{detail ?? status.detail}</p>
            <BenchmarkInfoSection title="Harness">
              <BenchmarkInfoRow label="Framework" value="EvalScope" />
              <BenchmarkInfoRow label="Dataset" value="mmmu_pro" />
              <BenchmarkInfoRow label="Metric" value="acc" />
              <BenchmarkInfoRow label="Status" value={status.statusLabel} />
              <BenchmarkInfoRow label="Vision model" value="Required" />
              <BenchmarkInfoRow label="MMPROJ" value="Required" />
              <BenchmarkInfoRow label="Shared harness" value={harnessInstalled ? "Installed" : "Needs harness"} />
              <BenchmarkInfoRow label="EvalScope" value={harnessStatus?.evalscope ?? "Unavailable"} />
            </BenchmarkInfoSection>
            <BenchmarkInfoSection title="MMMU-Pro Dataset">
              <BenchmarkInfoRow label="Downloaded" value={datasetStatus.datasetReady ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Verified" value={datasetStatus.datasetStatusLabel === "Verified" ? "Yes" : "No"} />
              <BenchmarkInfoRow label="Samples" value="1,730" />
              <BenchmarkInfoRow label="Official asset" value="AI-ModelScope/MMMU_Pro" />
              <BenchmarkInfoRow label="Cache path" value={datasetStatus.datasetPath ?? "Not downloaded"} />
            </BenchmarkInfoSection>
          </aside>
        </div>
      </div>
    </section>
  );
}

function BenchmarkInfoSection({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <section className="benchmark-info-section">
      <h2>{title}</h2>
      <div>{children}</div>
    </section>
  );
}

function BenchmarkInfoRow({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="benchmark-info-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function BenchmarkInputRow({
  label,
  inputLabel,
  value,
  placeholder,
  inputMode,
  onChange,
}: {
  label: string;
  inputLabel: string;
  value: string;
  placeholder: string;
  inputMode: "numeric" | "decimal";
  onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <label className="benchmark-info-row benchmark-input-row">
      <span>{label}</span>
      <input
        aria-label={inputLabel}
        className="benchmark-config-input"
        inputMode={inputMode}
        value={value}
        placeholder={placeholder}
        onChange={onChange}
      />
    </label>
  );
}

function BenchmarkSelectRow<T extends string>({
  label,
  selectLabel,
  value,
  onChange,
  options,
}: {
  label: string;
  selectLabel: string;
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string }[];
}) {
  const [open, setOpen] = useState(false);
  const selectedOption = options.find((option) => option.value === value) ?? options[0];

  return (
    <div className="benchmark-info-row benchmark-select-row">
      <span>{label}</span>
      <div
        className="benchmark-select-control"
        onBlur={(event) => {
          if (!event.currentTarget.contains(event.relatedTarget)) {
            setOpen(false);
          }
        }}
      >
        <button
          type="button"
          className="benchmark-select-button"
          aria-label={`${selectLabel} ${selectedOption.label}`}
          aria-haspopup="listbox"
          aria-expanded={open}
          onClick={() => setOpen((current) => !current)}
        >
          <span>{selectedOption.label}</span>
          <span className="codicon codicon-chevron-down" aria-hidden="true" />
        </button>
        {open && (
          <div className="benchmark-select-menu" role="listbox" aria-label={selectLabel}>
            {options.map((option) => (
              <button
                key={option.value}
                type="button"
                className="benchmark-select-option"
                role="option"
                aria-selected={option.value === value}
                onMouseDown={(event) => event.preventDefault()}
                onClick={() => {
                  onChange(option.value);
                  setOpen(false);
                }}
              >
                {option.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

