import { useCallback, useState } from "react";
import { TitleBar } from "./components/TitleBar";
import { WorkbenchShell } from "./components/Workbench/WorkbenchShell";
import {
  EVAL_RESULTS_TAB_ID,
  layerEditorTab,
  type EditorTab,
} from "./components/Workbench/editorTabModel";
import { useModel } from "./hooks/useModel";
import { useRecipe } from "./hooks/useRecipe";
import { useProgress } from "./hooks/useProgress";
import { isTauri } from "@tauri-apps/api/core";
import {
  testRecipe,
  cancelRecipeTest,
  saveRecipe,
  loadRecipe,
  exportGguf,
} from "./lib/tauri-bridge";
import type {
  BenchmarkResult,
  RecipeEvalPreset,
  RecipeState,
  RecipeTestMode,
} from "./types";
import { setMockInvoke } from "./lib/tauri-bridge";

// Auto-inject the mock bridge only in plain browser runs.
if (typeof window !== "undefined" && !isTauri()) {
  void import("../tests/mocks/tauri-bridge")
    .then(({ createMockBridge }) => setMockInvoke(createMockBridge()))
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

  const handleOpenModel = useCallback(async () => {
    let selected: string | null = null;

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
    if (!recipe) return;
    if (recipe.baseModel !== modelPath) {
      setAppError("Recipe model does not match the loaded model. Reload the model or recipe.");
      return;
    }
    startOperation();
    try {
      const result = await testRecipe(
        recipe,
        512,
        recipeTestMode,
        recipeEvalPreset,
      );
      setBenchmarkResult(result);
      setOpenEditors((current) =>
        current.some((editor) => editor.id === EVAL_RESULTS_TAB_ID)
          ? current
          : [...current, { id: EVAL_RESULTS_TAB_ID, kind: "eval-results" }],
      );
      setActiveEditorId(EVAL_RESULTS_TAB_ID);
      setAppError(null);
      setProfile({ vramEstimate: result.vramAllocatedMb, sizeSavedVsQ8: 0 });
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
    startOperation,
    endOperation,
    setProfile,
  ]);

  const handleCancelTest = useCallback(async () => {
    if (!running || cancelling) return;
    requestCancellation();
    try {
      await cancelRecipeTest();
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
          evalPreset={recipeEvalPreset}
          testMode={recipeTestMode}
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
