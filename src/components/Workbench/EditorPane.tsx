import {
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
  GpqaDiamondStatus,
  GpqaShotMode,
  GpqaThinkingMode,
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
  cancelling: boolean;
  progress: ProgressEvent | null;
  outputLines: BenchmarkOutputLine[];
  apiOutputLines: BenchmarkOutputLine[];
  openEditors: EditorTab[];
  activeEditorId: string | null;
  benchmarkResult: BenchmarkResult | null;
  tensors: TensorInfo[];
  assignments: Record<string, QuantType>;
  profile: RecipeProfile | null;
  evalPreset: RecipeEvalPreset;
  testMode: RecipeTestMode;
  selectedRunIds: BenchmarkRunId[];
  gpqaStatus: GpqaDiamondStatus;
  gpqaShotMode: GpqaShotMode;
  gpqaConfig: GpqaBenchmarkConfigInput;
  onSelectEditor: (editorId: string) => void;
  onCloseEditor: (editorId: string) => void;
  onReorderEditor: (editorId: string, beforeEditorId: string | null) => void;
  onAssignQuant: (tensorName: string, quantType: QuantType) => void;
  onInstallGpqaHarness: () => void;
  onDownloadGpqaDataset: () => void;
  onRefreshGpqaStatus: () => void;
  onEvalPresetChange: (preset: RecipeEvalPreset) => void;
  onTestModeChange: (mode: RecipeTestMode) => void;
  onToggleRunTarget: (target: BenchmarkRunId) => void;
  onGpqaShotModeChange: (mode: GpqaShotMode) => void;
  onGpqaConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onTest: () => void;
  onCancelTest: () => void;
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
  cancelling,
  progress,
  outputLines,
  apiOutputLines,
  openEditors,
  activeEditorId,
  benchmarkResult,
  tensors,
  assignments,
  profile,
  evalPreset,
  testMode,
  selectedRunIds,
  gpqaStatus,
  gpqaShotMode,
  gpqaConfig,
  onSelectEditor,
  onCloseEditor,
  onReorderEditor,
  onAssignQuant,
  onInstallGpqaHarness,
  onDownloadGpqaDataset,
  onRefreshGpqaStatus,
  onEvalPresetChange,
  onTestModeChange,
  onToggleRunTarget,
  onGpqaShotModeChange,
  onGpqaConfigChange,
  onTest,
  onCancelTest,
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
      : activeEditor?.kind === "gpqa-details"
        ? "GPQA Diamond Details"
        : activeEditor?.kind === "gpqa-dataset"
          ? "GPQA Diamond Dataset"
      : layerTitle(activeLayerIndex);
  const activeBreadcrumb = activeEditor ? editorTabLabel(activeEditor) : "workspace";
  const showingResults = activeEditor?.kind === "eval-results" && benchmarkResult;
  const showingGpqaDetails = activeEditor?.kind === "gpqa-details";
  const showingGpqaDataset = activeEditor?.kind === "gpqa-dataset";

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
          onEvalPresetChange={onEvalPresetChange}
          onTestModeChange={onTestModeChange}
          onToggleRunTarget={onToggleRunTarget}
          onTest={onTest}
          onCancelTest={onCancelTest}
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
      ) : showingGpqaDetails ? (
        <GpqaDetailsView
          status={gpqaStatus}
          shotMode={gpqaShotMode}
          config={gpqaConfig}
          running={running}
          onInstallHarness={onInstallGpqaHarness}
          onRefreshStatus={onRefreshGpqaStatus}
          onShotModeChange={onGpqaShotModeChange}
          onConfigChange={onGpqaConfigChange}
        />
      ) : showingGpqaDataset ? (
        <GpqaDatasetView
          status={gpqaStatus}
          running={running}
          onDownloadDataset={onDownloadGpqaDataset}
          onRefreshStatus={onRefreshGpqaStatus}
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
      <BottomPanel
        tensors={tensors}
        assignments={assignments}
        profile={profile}
        outputLines={outputLines}
        apiOutputLines={apiOutputLines}
      />
    </main>
  );
}

function GpqaDetailsView({
  status,
  shotMode,
  config,
  running,
  onInstallHarness,
  onRefreshStatus,
  onShotModeChange,
  onConfigChange,
}: {
  status: GpqaDiamondStatus;
  shotMode: GpqaShotMode;
  config: GpqaBenchmarkConfigInput;
  running: boolean;
  onInstallHarness: () => void;
  onRefreshStatus: () => void;
  onShotModeChange: (mode: GpqaShotMode) => void;
  onConfigChange: (config: GpqaBenchmarkConfigInput) => void;
}) {
  const harnessReady = status.python && status.evalscope;
  const updateIntegerField =
    (field: "contextWindow" | "sampleLimit") =>
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.currentTarget.value;
      if (/^\d*$/.test(value)) onConfigChange({ ...config, [field]: value });
    };
  const updateTemperature = (event: ChangeEvent<HTMLInputElement>) => {
    const value = event.currentTarget.value;
    if (/^\d*(?:\.\d*)?$/.test(value)) {
      onConfigChange({ ...config, temperature: value });
    }
  };
  const updateThinking = (thinking: GpqaThinkingMode) => {
    onConfigChange({ ...config, thinking });
  };

  return (
    <section className="benchmark-editor-surface">
      <div className="benchmark-editor-content">
        <div className="tensor-editor-title">
          <div>
            <h1>GPQA Diamond Details</h1>
          </div>
        </div>
        <BenchmarkInfoSection title="Status">
          <BenchmarkInfoRow label="Run state" value={status.statusLabel} />
          <BenchmarkInfoRow label="Readiness" value={status.ready ? "Ready" : "Not ready"} />
        </BenchmarkInfoSection>
        <BenchmarkInfoSection title="Harness">
          <BenchmarkInfoRow label="Framework" value="EvalScope" />
          <BenchmarkInfoRow label="Dataset" value="gpqa_diamond" />
          <BenchmarkInfoRow label="Metric" value="acc" />
          <BenchmarkInfoRow label="Status" value={harnessReady ? "Installed" : status.statusLabel} />
          <BenchmarkInfoRow label="Python" value={status.python ?? "Unavailable"} />
          <BenchmarkInfoRow label="EvalScope" value={status.evalscope ?? "Unavailable"} />
        </BenchmarkInfoSection>
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
            onChange={updateTemperature}
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
        <BenchmarkInfoSection title="Comparability">
          <BenchmarkInfoRow
            label="Selected mode"
            value={shotMode === "five_shot_cot" ? "5-shot CoT" : "0-shot CoT"}
          />
          <BenchmarkInfoRow label="Formal run" value="Full 198 sample dataset" />
        </BenchmarkInfoSection>
        <BenchmarkActions>
          <button
            type="button"
            className="benchmark-action-button"
            disabled={running || Boolean(harnessReady)}
            onClick={onInstallHarness}
          >
            Install harness
          </button>
          <button
            type="button"
            className="benchmark-action-button secondary"
            disabled={running}
            onClick={onRefreshStatus}
          >
            Refresh
          </button>
        </BenchmarkActions>
        <p className="benchmark-detail-text">{status.detail}</p>
      </div>
    </section>
  );
}

function GpqaDatasetView({
  status,
  running,
  onDownloadDataset,
  onRefreshStatus,
}: {
  status: GpqaDiamondStatus;
  running: boolean;
  onDownloadDataset: () => void;
  onRefreshStatus: () => void;
}) {
  const harnessReady = status.python && status.evalscope;

  return (
    <section className="benchmark-editor-surface">
      <div className="benchmark-editor-content">
        <div className="tensor-editor-title">
          <div>
            <h1>GPQA Diamond Dataset</h1>
          </div>
        </div>
        <BenchmarkInfoSection title="Status">
          <BenchmarkInfoRow label="Downloaded" value={status.datasetPath ? "Yes" : "No"} />
          <BenchmarkInfoRow label="Verified" value={status.datasetReady ? "Yes" : "No"} />
          <BenchmarkInfoRow label="Samples" value="198" />
          <BenchmarkInfoRow label="License" value="CC-BY-4.0" />
        </BenchmarkInfoSection>
        <BenchmarkInfoSection title="Source">
          <BenchmarkInfoRow label="Official asset" value={status.datasetUrl} />
          <BenchmarkInfoRow label="Cache path" value={status.datasetPath ?? "Not downloaded"} />
          <BenchmarkInfoRow label="SHA256" value={status.datasetHash ?? "Unavailable"} />
          <BenchmarkInfoRow label="Expected SHA256" value={status.expectedDatasetHash} />
        </BenchmarkInfoSection>
        <BenchmarkActions>
          <button
            type="button"
            className="benchmark-action-button"
            disabled={running || status.datasetReady || !harnessReady}
            onClick={onDownloadDataset}
          >
            Download dataset
          </button>
          <button
            type="button"
            className="benchmark-action-button secondary"
            disabled={running}
            onClick={onRefreshStatus}
          >
            Verify hash
          </button>
        </BenchmarkActions>
        <p className="benchmark-detail-text">
          Dataset status: {status.datasetStatusLabel}. {status.detail}
        </p>
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

function BenchmarkActions({ children }: { children: ReactNode }) {
  return <div className="benchmark-actions">{children}</div>;
}
