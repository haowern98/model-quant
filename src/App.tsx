import { useCallback, useEffect, useState } from "react";
import { TitleBar } from "./components/TitleBar";
import { WorkbenchShell } from "./components/Workbench/WorkbenchShell";
import {
  EVAL_RESULTS_TAB_ID,
  gpqaDatasetEditorTab,
  gpqaDetailsEditorTab,
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
  GpqaDiamondStatus,
  GpqaShotMode,
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
  const { outputLines } = useBenchmarkOutputLog();

  const [openEditors, setOpenEditors] = useState<EditorTab[]>([]);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);
  const [expandedLayers, setExpandedLayers] = useState<Set<number>>(
    () => new Set(),
  );
  const [benchmarkResult, setBenchmarkResult] =
    useState<BenchmarkResult | null>(null);
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
  const [gpqaShotMode, setGpqaShotMode] =
    useState<GpqaShotMode>("five_shot_cot");

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

  const resetForLoadedModel = useCallback(
    (path: string, loadedModel: NonNullable<typeof model>) => {
      const tensors = loadedModel.tensors.map((t) => ({
        name: t.name,
        currentQuant: t.currentQuant,
      }));
      resetRecipeForModel(path, tensors);
      setOpenEditors([]);
      setActiveEditorId(null);
      setExpandedLayers(new Set());
      setBenchmarkResult(null);
      setAppError(null);
    },
    [resetRecipeForModel],
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
    setBenchmarkResult(null);
    handleCloseEditor(EVAL_RESULTS_TAB_ID);
  }, [handleCloseEditor]);

  const openEvalResults = useCallback((result: BenchmarkResult) => {
    setBenchmarkResult(result);
    setOpenEditors((current) =>
      current.some((editor) => editor.id === EVAL_RESULTS_TAB_ID)
        ? current
        : [...current, { id: EVAL_RESULTS_TAB_ID, kind: "eval-results" }],
    );
    setActiveEditorId(EVAL_RESULTS_TAB_ID);
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
        setAppError("Open a GGUF model before running GPQA Diamond.");
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
        const apiStatus = await startModelInspectorApi({
          benchmarkLabel: "ModelInspector API",
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
    startOperation,
    endOperation,
    openEvalResults,
    setProfile,
  ]);

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

  return (
    <div className="app-root">
      <TitleBar modelPath={modelPath} onOpenModel={handleOpenModel} />
      <div className="app-body">
        {(modelError || appError) && (
          <div className="app-error" role="alert">
            {modelError ?? appError}
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
          benchmarkResult={benchmarkResult}
          expandedLayers={expandedLayers}
          running={running}
          cancelling={cancelling}
          progress={progress}
          outputLines={outputLines}
          evalPreset={recipeEvalPreset}
          testMode={recipeTestMode}
          selectedRunIds={selectedRunIds}
          gpqaStatus={gpqaStatus}
          gpqaShotMode={gpqaShotMode}
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
          onGpqaShotModeChange={setGpqaShotMode}
          onInstallGpqaHarness={handleInstallGpqaHarness}
          onDownloadGpqaDataset={handleDownloadGpqaDataset}
          onRefreshGpqaStatus={refreshGpqaStatus}
          onOpenGpqaDetails={handleOpenGpqaDetails}
          onOpenGpqaDataset={handleOpenGpqaDataset}
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
