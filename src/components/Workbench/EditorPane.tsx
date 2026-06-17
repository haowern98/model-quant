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
type GpqaBenchmarkTab = "details" | "dataset" | "configuration";

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
        ? "GPQA Diamond"
        : activeEditor?.kind === "gpqa-dataset"
          ? "GPQA Diamond Dataset"
      : layerTitle(activeLayerIndex);
  const activeBreadcrumb = activeEditor ? editorTabLabel(activeEditor) : "workspace";
  const showingResults = activeEditor?.kind === "eval-results" && benchmarkResult;
  const showingGpqaDetails = activeEditor?.kind === "gpqa-details";
  const showingGpqaDataset = activeEditor?.kind === "gpqa-dataset";
  const showingGpqaBenchmark = showingGpqaDetails || showingGpqaDataset;

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
        <span>{showingResults || showingGpqaBenchmark ? "benchmark" : "tensors"}</span>
      </div>

      {showingResults ? (
        <EvalResultsView
          result={benchmarkResult}
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
          onRefreshStatus={onRefreshGpqaStatus}
          onShotModeChange={onGpqaShotModeChange}
          onConfigChange={onGpqaConfigChange}
          onRunBenchmark={onTest}
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

function GpqaBenchmarkView({
  initialTab,
  status,
  shotMode,
  config,
  running,
  onInstallHarness,
  onDownloadDataset,
  onRefreshStatus,
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
  onRefreshStatus: () => void;
  onShotModeChange: (mode: GpqaShotMode) => void;
  onConfigChange: (config: GpqaBenchmarkConfigInput) => void;
  onRunBenchmark: () => void;
}) {
  const [activeTab, setActiveTab] = useState<GpqaBenchmarkTab>(initialTab);
  const harnessReady = status.python && status.evalscope;
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
                <button
                  type="button"
                  className="benchmark-action-button secondary"
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
                <button
                  type="button"
                  className="benchmark-action-button primary"
                  disabled={running || !status.ready}
                  onClick={onRunBenchmark}
                >
                  Run Benchmark
                </button>
                <button type="button" className="benchmark-icon-button" aria-label="GPQA settings">
                  <span className="codicon codicon-settings-gear" aria-hidden="true" />
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
                <BenchmarkInfoSection title="Sample Row">
                  <BenchmarkInfoRow
                    label="Question"
                    value="Which energy difference allows two quantum states to be clearly resolved?"
                  />
                  <BenchmarkInfoRow label="Choices" value="A, B, C, D multiple-choice answer set" />
                  <BenchmarkInfoRow label="Expected output" value="ANSWER: A-D" />
                </BenchmarkInfoSection>
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

