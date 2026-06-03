import { useCallback, useState } from "react";
import { TitleBar } from "./components/TitleBar";
import { WorkbenchShell } from "./components/Workbench/WorkbenchShell";
import { TestResultsModal } from "./components/TestResultsModal/TestResultsModal";
import { useModel } from "./hooks/useModel";
import { useRecipe } from "./hooks/useRecipe";
import { useProgress } from "./hooks/useProgress";
import { isTauri } from "@tauri-apps/api/core";
import {
  testRecipe,
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
  const { progress, running, startOperation, endOperation } = useProgress();

  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number | null>(
    null,
  );
  const [openLayers, setOpenLayers] = useState<number[]>([]);
  const [expandedLayers, setExpandedLayers] = useState<Set<number>>(
    () => new Set(),
  );
  const [benchmarkResult, setBenchmarkResult] =
    useState<BenchmarkResult | null>(null);
  const [showResults, setShowResults] = useState(false);
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
      setSelectedLayerIndex(null);
      setOpenLayers([]);
      setExpandedLayers(new Set());
      setBenchmarkResult(null);
      setShowResults(false);
      setAppError(null);
    },
    [resetRecipeForModel],
  );

  const handleOpenLayer = useCallback((layerIndex: number) => {
    setSelectedLayerIndex(layerIndex);
    setOpenLayers((current) =>
      current.includes(layerIndex) ? current : [...current, layerIndex],
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

  const handleCloseLayer = useCallback((layerIndex: number) => {
    setOpenLayers((current) => {
      const next = current.filter((item) => item !== layerIndex);
      setSelectedLayerIndex((selected) => {
        if (selected !== layerIndex) return selected;
        return next.length > 0 ? next[next.length - 1] : null;
      });
      return next;
    });
  }, []);

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
      setShowResults(true);
      setAppError(null);
      setProfile({ vramEstimate: result.vramAllocatedMb, sizeSavedVsQ8: 0 });
    } catch (e) {
      setAppError((e as Error).message);
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
          openLayers={openLayers}
          expandedLayers={expandedLayers}
          running={running}
          progress={progress}
          evalPreset={recipeEvalPreset}
          testMode={recipeTestMode}
          onOpenLayer={handleOpenLayer}
          onToggleLayer={handleToggleLayer}
          onCloseLayer={handleCloseLayer}
          onAssignQuant={assignQuant}
          onAssignByPattern={assignByPattern}
          onEvalPresetChange={setRecipeEvalPreset}
          onTestModeChange={setRecipeTestMode}
          onTest={handleTest}
          onSaveRecipe={handleSaveRecipe}
          onLoadRecipe={handleLoadRecipe}
          onExport={handleExport}
        />
      </div>

      {showResults && (
        <TestResultsModal
          result={benchmarkResult}
          onSave={handleSaveRecipe}
          onExport={handleExport}
          onDiscard={() => setShowResults(false)}
        />
      )}
    </div>
  );
}

export default App;
