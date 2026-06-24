import { useCallback, useEffect, useState } from "react";
import { TitleBar } from "./components/TitleBar";
import { WorkbenchShell } from "./components/Workbench/WorkbenchShell";
import {
  evalResultsEditorTab,
  gpqaDatasetEditorTab,
  gpqaDetailsEditorTab,
  humanevalDetailsEditorTab,
  layerEditorTab,
  type EditorTab,
} from "./components/Workbench/editorTabModel";
import { useModel } from "./hooks/useModel";
import { useRecipe } from "./hooks/useRecipe";
import { useProgress } from "./hooks/useProgress";
import { useBenchmarkOutputLog } from "./hooks/useBenchmarkOutputLog";
import {
  testRecipe,
  cancelRecipeTest,
  cancelOfficialBenchmark,
  getGpqaDiamondStatus,
  getHumanEvalStatus,
  installGpqaDiamondHarness,
  downloadGpqaDiamondDataset,
  runGpqaDiamondBenchmark,
  saveRecipe,
  loadRecipe,
  exportGguf,
  startModelInspectorApi,
  stopModelInspectorApi,
} from "./lib/tauri-bridge";
import type {
  BenchmarkRunId,
  BenchmarkResult,
  GpqaBenchmarkConfig,
  GpqaBenchmarkConfigInput,
  GpqaDiamondStatus,
  GpqaShotMode,
  HumanEvalStatus,
  RecipeEvalPreset,
  RecipeState,
  RecipeTestMode,
} from "./types";
import { setMockInvoke } from "./lib/tauri-bridge";

const DEFAULT_GPQA_STATUS: GpqaDiamondStatus = {
  ready: false,
  statusLabel: "Needs harness",
  python: null,
  evalscope: null,
  datasetReady: false,
  datasetStatusLabel: "Missing",
  datasetPath: null,
  datasetHash: null,
  datasetUrl: "AI-ModelScope/gpqa_diamond",
  expectedDatasetHash: "EvalScope dataset cache marker",
  detail: "GPQA Diamond readiness has not been checked yet.",
};

const DEFAULT_HUMANEVAL_STATUS: HumanEvalStatus = {
  ready: false,
  statusLabel: "Needs Docker",
  python: null,
  evalscope: null,
  dockerReady: false,
  docker: null,
  detail: "HumanEval readiness has not been checked yet.",
};

const GPQA_DEFAULT_CONTEXT_WINDOW = 20_000;
const GPQA_SAMPLE_COUNT = 198;
const GPQA_DEFAULT_TEMPERATURE = 0;

const DEFAULT_GPQA_CONFIG_INPUT: GpqaBenchmarkConfigInput = {
  contextWindow: "",
  sampleLimit: "",
  temperature: "0",
  thinking: "off",
  topK: "40",
  repeatPenalty: "1.1",
  presencePenalty: "0",
  topP: "0.95",
  minP: "0.05",
};

function parseOptionalIntegerField(
  value: string,
  defaultValue: number,
  min: number,
  max: number,
  label: string,
): number | string {
  const trimmed = value.trim();
  if (trimmed === "") return defaultValue;
  if (!/^\d+$/.test(trimmed)) return `${label} must be a whole number.`;
  const parsed = Number(trimmed);
  if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    return `${label} must be between ${min} and ${max}.`;
  }
  return parsed;
}

function parseOptionalIntegerOverride(
  value: string,
  min: number,
  max: number,
  label: string,
): number | undefined | string {
  const trimmed = value.trim();
  if (trimmed === "") return undefined;
  if (!/^\d+$/.test(trimmed)) return `${label} must be a whole number.`;
  const parsed = Number(trimmed);
  if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    return `${label} must be between ${min} and ${max}.`;
  }
  return parsed;
}

function parseOptionalNumberOverride(
  value: string,
  min: number,
  max: number,
  label: string,
): number | undefined | string {
  const trimmed = value.trim();
  if (trimmed === "") return undefined;
  const parsed = Number(trimmed);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    return `${label} must be between ${min} and ${max}.`;
  }
  return parsed;
}

function resolveGpqaConfigInput(
  input: GpqaBenchmarkConfigInput,
): GpqaBenchmarkConfig | string {
  const contextWindow = parseOptionalIntegerField(
    input.contextWindow,
    GPQA_DEFAULT_CONTEXT_WINDOW,
    1,
    Number.MAX_SAFE_INTEGER,
    "GPQA context window",
  );
  if (typeof contextWindow === "string") return contextWindow;

  const sampleLimit = parseOptionalIntegerField(
    input.sampleLimit,
    GPQA_SAMPLE_COUNT,
    1,
    GPQA_SAMPLE_COUNT,
    "GPQA sample limit",
  );
  if (typeof sampleLimit === "string") return sampleLimit;

  const temperatureText = input.temperature.trim();
  const temperature =
    temperatureText === "" ? GPQA_DEFAULT_TEMPERATURE : Number(temperatureText);
  if (!Number.isFinite(temperature) || temperature < 0 || temperature > 2) {
    return "GPQA temperature must be between 0 and 2.";
  }

  const topK = parseOptionalIntegerOverride(input.topK, 0, 1000, "GPQA top K sampling");
  if (typeof topK === "string") return topK;

  const repeatPenalty = parseOptionalNumberOverride(
    input.repeatPenalty,
    0,
    3,
    "GPQA repeat penalty",
  );
  if (typeof repeatPenalty === "string") return repeatPenalty;

  const presencePenalty = parseOptionalNumberOverride(
    input.presencePenalty,
    -2,
    2,
    "GPQA presence penalty",
  );
  if (typeof presencePenalty === "string") return presencePenalty;

  const topP = parseOptionalNumberOverride(input.topP, 0, 1, "GPQA top P sampling");
  if (typeof topP === "string") return topP;

  const minP = parseOptionalNumberOverride(input.minP, 0, 1, "GPQA min P sampling");
  if (typeof minP === "string") return minP;

  return {
    contextWindow,
    sampleLimit,
    temperature,
    thinking: input.thinking,
    topK,
    repeatPenalty,
    presencePenalty,
    topP,
    minP,
  };
}

function hasTauriRuntime(): boolean {
  return typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
}

// Auto-inject the mock bridge only in plain browser runs.
if (typeof window !== "undefined" && !hasTauriRuntime()) {
  void import("../tests/mocks/tauri-bridge")
    .then(({ createMockBridge }) => {
      setMockInvoke(createMockBridge());
      (window as typeof window & { __MODEL_SURGERY_MOCK_READY__?: boolean })
        .__MODEL_SURGERY_MOCK_READY__ = true;
    })
    .catch(() => undefined);
}

function App() {
  const {
    model,
    modelPath,
    error: modelError,
    openModel,
    getTensorsForLayer,
  } = useModel();
  const {
    recipe,
    resetRecipeForModel,
    setRecipeState,
    assignQuant,
    assignByPattern,
    setProfile,
    getAssignments,
  } = useRecipe();
  const {
    progress,
    running,
    cancelling,
    startOperation,
    requestCancellation,
    endOperation,
  } = useProgress();
  const { outputLines, apiOutputLines } = useBenchmarkOutputLog();

  const [openEditors, setOpenEditors] = useState<EditorTab[]>([]);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);
  const [modelExplorerFocusVersion, setModelExplorerFocusVersion] = useState(0);
  const [expandedLayers, setExpandedLayers] = useState<Set<number>>(
    () => new Set(),
  );
  const [appError, setAppError] = useState<string | null>(null);
  const [recipeTestMode, setRecipeTestMode] =
    useState<RecipeTestMode>("single");
  const [recipeEvalPreset, setRecipeEvalPreset] =
    useState<RecipeEvalPreset>("default");
  const [selectedRunIds, setSelectedRunIds] = useState<BenchmarkRunId[]>([
    "ppl_check",
  ]);
  const [gpqaStatus, setGpqaStatus] =
    useState<GpqaDiamondStatus>(DEFAULT_GPQA_STATUS);
  const [humanevalStatus, setHumanEvalStatus] = useState<HumanEvalStatus>(
    DEFAULT_HUMANEVAL_STATUS,
  );
  const [gpqaShotMode, setGpqaShotMode] =
    useState<GpqaShotMode>("five_shot_cot");
  const [gpqaConfig, setGpqaConfig] = useState<GpqaBenchmarkConfigInput>(
    DEFAULT_GPQA_CONFIG_INPUT,
  );

  const refreshGpqaStatus = useCallback(() => {
    let cancelled = false;
    getGpqaDiamondStatus()
      .then((status) => {
        if (!cancelled) setGpqaStatus(status);
      })
      .catch((error) => {
        if (!cancelled) {
          setGpqaStatus({
            ...DEFAULT_GPQA_STATUS,
            detail: (error as Error).message,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => refreshGpqaStatus(), [refreshGpqaStatus, modelPath]);

  const refreshHumanEvalStatus = useCallback(() => {
    let cancelled = false;
    getHumanEvalStatus()
      .then((status) => {
        if (!cancelled) setHumanEvalStatus(status);
      })
      .catch((error) => {
        if (!cancelled) {
          setHumanEvalStatus({
            ...DEFAULT_HUMANEVAL_STATUS,
            detail: (error as Error).message,
          });
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => refreshHumanEvalStatus(), [refreshHumanEvalStatus]);

  useEffect(() => {
    const refreshBenchmarkStatuses = () => {
      void refreshGpqaStatus();
      void refreshHumanEvalStatus();
    };
    window.addEventListener("modelinspector:benchmark-harness-changed", refreshBenchmarkStatuses);
    return () => {
      window.removeEventListener("modelinspector:benchmark-harness-changed", refreshBenchmarkStatuses);
    };
  }, [refreshGpqaStatus, refreshHumanEvalStatus]);

  const resetForLoadedModel = useCallback(
    (path: string, loadedModel: NonNullable<typeof model>) => {
      const tensors = loadedModel.tensors.map((t) => ({
        name: t.name,
        currentQuant: t.currentQuant,
      }));
      resetRecipeForModel(path, tensors);
      setOpenEditors((current) => current.filter((editor) => editor.kind !== "layer"));
      setActiveEditorId((active) => {
        if (!active) return null;
        const activeEditor = openEditors.find((editor) => editor.id === active);
        if (activeEditor?.kind !== "layer") return active;
        return openEditors.find((editor) => editor.kind !== "layer")?.id ?? null;
      });
      setModelExplorerFocusVersion((version) => version + 1);
      setExpandedLayers(new Set());
      setAppError(null);
    },
    [openEditors, resetRecipeForModel],
  );

  const handleOpenLayer = useCallback((layerIndex: number) => {
    const tab = layerEditorTab(layerIndex);
    setActiveEditorId(tab.id);
    setOpenEditors((current) =>
      current.some((editor) => editor.id === tab.id) ? current : [...current, tab],
    );
    setExpandedLayers((current) => {
      const next = new Set(current);
      next.add(layerIndex);
      return next;
    });
  }, []);

  const handleToggleLayer = useCallback((layerIndex: number) => {
    setExpandedLayers((current) => {
      const next = new Set(current);
      if (next.has(layerIndex)) next.delete(layerIndex);
      else next.add(layerIndex);
      return next;
    });
  }, []);

  const handleCloseEditor = useCallback((editorId: string) => {
    setOpenEditors((current) => {
      const next = current.filter((editor) => editor.id !== editorId);
      setActiveEditorId((active) => {
        if (active !== editorId) return active;
        return next.length > 0 ? next[next.length - 1].id : null;
      });
      return next;
    });
  }, []);

  const handleReorderEditor = useCallback((editorId: string, beforeEditorId: string | null) => {
    setOpenEditors((current) => {
      const moving = current.find((editor) => editor.id === editorId);
      if (!moving) return current;

      const remaining = current.filter((editor) => editor.id !== editorId);
      const insertIndex =
        beforeEditorId === null
          ? remaining.length
          : remaining.findIndex((editor) => editor.id === beforeEditorId);
      if (insertIndex < 0) return current;

      const next = [...remaining];
      next.splice(insertIndex, 0, moving);
      return next;
    });
  }, []);

  const handleDiscardResults = useCallback(() => {
    if (!activeEditorId) return;
    const activeEditor = openEditors.find((editor) => editor.id === activeEditorId);
    if (activeEditor?.kind === "eval-results") handleCloseEditor(activeEditorId);
  }, [activeEditorId, handleCloseEditor, openEditors]);

  const openEvalResults = useCallback((result: BenchmarkResult) => {
    const tab = evalResultsEditorTab(result);
    setOpenEditors((current) => [...current, tab]);
    setActiveEditorId(tab.id);
  }, []);

  const openEditorTab = useCallback((tab: EditorTab) => {
    setOpenEditors((current) =>
      current.some((editor) => editor.id === tab.id) ? current : [...current, tab],
    );
    setActiveEditorId(tab.id);
  }, []);

  const handleOpenGpqaDetails = useCallback(() => {
    openEditorTab(gpqaDetailsEditorTab());
  }, [openEditorTab]);

  const handleOpenGpqaDataset = useCallback(() => {
    openEditorTab(gpqaDatasetEditorTab());
  }, [openEditorTab]);

  const handleOpenHumanEvalDetails = useCallback(() => {
    openEditorTab(humanevalDetailsEditorTab());
  }, [openEditorTab]);

  const handleToggleRunTarget = useCallback(
    (target: BenchmarkRunId) => {
      if (target !== "ppl_check" && target !== "gpqa_diamond") return;
      setSelectedRunIds((current) =>
        current.includes(target)
          ? current.filter((id) => id !== target)
          : [...current, target],
      );
    },
    [],
  );

  const handleOpenModel = useCallback(async () => {
    let selected: string | null = null;

    if (!hasTauriRuntime()) {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".gguf";
      input.style.display = "none";
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
          const loadedModel = await openModel(file.name);
          if (loadedModel) resetForLoadedModel(file.name, loadedModel);
        }
        input.remove();
      };
      document.body.appendChild(input);
      input.click();
      return;
    }

    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const dialogResult = await open({
        filters: [{ name: "GGUF", extensions: ["gguf"] }],
      });
      if (dialogResult && typeof dialogResult === "string")
        selected = dialogResult;
    } catch {
      const input = document.createElement("input");
      input.type = "file";
      input.accept = ".gguf";
      input.style.display = "none";
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) {
          const loadedModel = await openModel(file.name);
          if (loadedModel) resetForLoadedModel(file.name, loadedModel);
        }
        input.remove();
      };
      document.body.appendChild(input);
      input.click();
      return;
    }

    if (selected) {
      const loadedModel = await openModel(selected);
      if (loadedModel) resetForLoadedModel(selected, loadedModel);
    }
  }, [openModel, resetForLoadedModel]);

  const handleTest = useCallback(async () => {
    if (!recipe || !modelPath) {
      if (
        selectedRunIds.includes("gpqa_diamond") &&
        !selectedRunIds.includes("ppl_check")
      ) {
        setAppError("Open a GGUF model before running tests.");
      } else if (selectedRunIds.some((id) => id === "ppl_check" || id === "gpqa_diamond")) {
        setAppError("Open a GGUF model before running selected tests.");
      }
      return;
    }
    if (recipe.baseModel !== modelPath) {
      setAppError("Recipe model does not match the loaded model. Reload the model or recipe.");
      return;
    }
    const runQueue = selectedRunIds.filter(
      (id) => id === "ppl_check" || (id === "gpqa_diamond" && gpqaStatus.ready),
    );
    if (runQueue.length === 0) {
      if (selectedRunIds.includes("gpqa_diamond") && !gpqaStatus.ready) {
        setAppError(`GPQA Diamond is not ready. Current status: ${gpqaStatus.statusLabel}.`);
      }
      return;
    }

    startOperation();
    try {
      let latestResult: BenchmarkResult | null = null;

      if (runQueue.includes("ppl_check")) {
        latestResult = await testRecipe(
          recipe,
          512,
          recipeTestMode,
          recipeEvalPreset,
        );
        openEvalResults(latestResult);
        setProfile({ vramEstimate: latestResult.vramAllocatedMb, sizeSavedVsQ8: 0 });
      }

      if (runQueue.includes("gpqa_diamond")) {
        const resolvedGpqaConfig = resolveGpqaConfigInput(gpqaConfig);
        if (typeof resolvedGpqaConfig === "string") {
          throw new Error(resolvedGpqaConfig);
        }
        const apiStatus = await startModelInspectorApi({
          benchmarkLabel: "ModelInspector API",
          contextWindow: resolvedGpqaConfig.contextWindow,
          defaultEnableThinking: resolvedGpqaConfig.thinking === "on",
        });
        if (!apiStatus.baseUrl || !apiStatus.apiKey || !apiStatus.modelId) {
          throw new Error("ModelInspector API did not return a usable benchmark endpoint.");
        }
        try {
          latestResult = await runGpqaDiamondBenchmark(
            apiStatus.baseUrl,
            apiStatus.apiKey,
            apiStatus.modelId,
            gpqaShotMode,
            resolvedGpqaConfig,
          );
          openEvalResults(latestResult);
        } finally {
          await stopModelInspectorApi();
        }
      }

      setAppError(null);
    } catch (e) {
      const message = (e as Error).message;
      if (!message.toLowerCase().includes("cancelled")) setAppError(message);
    } finally {
      endOperation();
    }
  }, [
    recipe,
    modelPath,
    recipeTestMode,
    recipeEvalPreset,
    selectedRunIds,
    gpqaStatus.ready,
    gpqaShotMode,
    gpqaConfig,
    startOperation,
    endOperation,
    openEvalResults,
    setProfile,
  ]);

  const handleNoTestsSelected = useCallback(() => {
    setAppError("Select at least one test before running.");
  }, []);

  const handleInstallGpqaHarness = useCallback(async () => {
    startOperation();
    try {
      const status = await installGpqaDiamondHarness();
      setGpqaStatus(status);
      setAppError(null);
    } catch (e) {
      setAppError((e as Error).message);
      void refreshGpqaStatus();
    } finally {
      endOperation();
    }
  }, [endOperation, refreshGpqaStatus, startOperation]);

  const handleDownloadGpqaDataset = useCallback(async () => {
    startOperation();
    try {
      const status = await downloadGpqaDiamondDataset();
      setGpqaStatus(status);
      setAppError(null);
    } catch (e) {
      setAppError((e as Error).message);
      void refreshGpqaStatus();
    } finally {
      endOperation();
    }
  }, [endOperation, refreshGpqaStatus, startOperation]);

  const handleCancelTest = useCallback(async () => {
    if (!running || cancelling) return;
    requestCancellation();
    try {
      await cancelRecipeTest();
      await cancelOfficialBenchmark();
      await stopModelInspectorApi();
    } catch (e) {
      setAppError((e as Error).message);
    }
  }, [running, cancelling, requestCancellation]);

  const handleSaveRecipe = useCallback(async () => {
    if (!recipe) return;
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const path = await save({
        filters: [{ name: "Recipe JSON", extensions: ["json"] }],
      });
      if (path && typeof path === "string") await saveRecipe(path, recipe);
    } catch {
      /* browser fallback: no-op */
    }
  }, [recipe]);

  const handleExport = useCallback(async () => {
    if (!recipe) return;
    try {
      const { save } = await import("@tauri-apps/plugin-dialog");
      const path = await save({
        filters: [{ name: "GGUF", extensions: ["gguf"] }],
      });
      if (path && typeof path === "string") await exportGguf(path, recipe);
    } catch {
      /* browser fallback: no-op */
    }
  }, [recipe]);

  const handleLoadRecipe = useCallback(async () => {
    try {
      const { open } = await import("@tauri-apps/plugin-dialog");
      const selected = await open({
        filters: [{ name: "Recipe JSON", extensions: ["json"] }],
      });
      if (selected && typeof selected === "string") {
        const loaded: RecipeState = await loadRecipe(selected);
        setRecipeState(loaded);
        setAppError(null);
      }
    } catch (e) {
      setAppError((e as Error).message);
    }
  }, [setRecipeState]);

  const activeEditor =
    openEditors.find((editor) => editor.id === activeEditorId) ?? null;
  const selectedLayerIndex =
    activeEditor?.kind === "layer" ? activeEditor.layerIndex : null;
  const selectedTensors =
    selectedLayerIndex !== null ? getTensorsForLayer(selectedLayerIndex) : [];
  const latestBenchmarkResult =
    [...openEditors]
      .reverse()
      .find(
        (editor): editor is Extract<EditorTab, { kind: "eval-results" }> =>
          editor.kind === "eval-results",
      )?.result ?? null;
  const visibleError = modelError ?? appError;

  return (
    <div className="app-root">
      <TitleBar modelPath={modelPath} onOpenModel={handleOpenModel} />
      <div className="app-body">
        {visibleError && (
          <div className="app-error-toast" role="alert">
            <span className="codicon codicon-error app-error-toast-icon" aria-hidden="true" />
            <span className="app-error-toast-message">{visibleError}</span>
            <button
              type="button"
              className="app-error-toast-close"
              aria-label="Dismiss error"
              onClick={() => {
                setAppError(null);
              }}
            >
              <span className="codicon codicon-close" aria-hidden="true" />
            </button>
          </div>
        )}
        <WorkbenchShell
          modelPath={modelPath}
          tensors={model?.tensors ?? []}
          selectedTensors={selectedTensors}
          assignments={getAssignments}
          profile={recipe?.profile ?? null}
          activeLayerIndex={selectedLayerIndex}
          openEditors={openEditors}
          activeEditorId={activeEditorId}
          latestBenchmarkResult={latestBenchmarkResult}
          expandedLayers={expandedLayers}
          running={running}
          cancelling={cancelling}
          progress={progress}
          outputLines={outputLines}
          apiOutputLines={apiOutputLines}
          evalPreset={recipeEvalPreset}
          testMode={recipeTestMode}
          selectedRunIds={selectedRunIds}
          gpqaStatus={gpqaStatus}
          humanevalStatus={humanevalStatus}
          gpqaShotMode={gpqaShotMode}
          gpqaConfig={gpqaConfig}
          modelExplorerFocusVersion={modelExplorerFocusVersion}
          onOpenLayer={handleOpenLayer}
          onOpenModel={handleOpenModel}
          onToggleLayer={handleToggleLayer}
          onSelectEditor={setActiveEditorId}
          onCloseEditor={handleCloseEditor}
          onReorderEditor={handleReorderEditor}
          onAssignQuant={assignQuant}
          onAssignByPattern={assignByPattern}
          onEvalPresetChange={setRecipeEvalPreset}
          onTestModeChange={setRecipeTestMode}
          onToggleRunTarget={handleToggleRunTarget}
          onNoTestsSelected={handleNoTestsSelected}
          onGpqaShotModeChange={setGpqaShotMode}
          onGpqaConfigChange={setGpqaConfig}
          onInstallGpqaHarness={handleInstallGpqaHarness}
          onDownloadGpqaDataset={handleDownloadGpqaDataset}
          onRefreshGpqaStatus={refreshGpqaStatus}
          onOpenGpqaDetails={handleOpenGpqaDetails}
          onOpenGpqaDataset={handleOpenGpqaDataset}
          onOpenHumanEvalDetails={handleOpenHumanEvalDetails}
          onTest={handleTest}
          onCancelTest={handleCancelTest}
          onSaveRecipe={handleSaveRecipe}
          onLoadRecipe={handleLoadRecipe}
          onExport={handleExport}
          onDiscardResults={handleDiscardResults}
        />
      </div>
    </div>
  );
}

export default App;
