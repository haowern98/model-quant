import { useCallback, useEffect, useState } from 'react';
import { TitleBar } from './components/TitleBar';
import { AppShell } from './components/AppShell';
import { Toolbar } from './components/Toolbar/Toolbar';
import { LayerBrowser } from './components/LayerBrowser/LayerBrowser';
import { BulkAssignPanel } from './components/LayerBrowser/BulkAssignPanel';
import { DetailPanel } from './components/DetailPanel/DetailPanel';
import { TestResultsModal } from './components/TestResultsModal/TestResultsModal';
import { useModel } from './hooks/useModel';
import { useRecipe } from './hooks/useRecipe';
import { useProgress } from './hooks/useProgress';
import { isTauri } from '@tauri-apps/api/core';
import { testRecipe, saveRecipe, loadRecipe, exportGguf } from './lib/tauri-bridge';
import type { BenchmarkResult, RecipeState } from './types';
import { toTargetQuant } from './types';
import { setMockInvoke } from './lib/tauri-bridge';

// Auto-inject the mock bridge only in plain browser runs.
if (typeof window !== 'undefined' && !isTauri()) {
  void import('../tests/mocks/tauri-bridge')
    .then(({ createMockBridge }) => setMockInvoke(createMockBridge()))
    .catch(() => undefined);
}

function App() {
  const { model, modelPath, error: modelError, openModel, getTensorsForLayer } = useModel();
  const { recipe, initRecipe, setRecipeState, assignQuant, assignAll, assignByPattern, setProfile, getAssignments } = useRecipe();
  const { progress, running, startOperation, endOperation } = useProgress();

  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const [showResults, setShowResults] = useState(false);

  const handleOpenModel = useCallback(async () => {
    let selected: string | null = null;

    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const dialogResult = await open({
        filters: [{ name: 'GGUF', extensions: ['gguf'] }],
      });
      if (dialogResult && typeof dialogResult === 'string') selected = dialogResult;
    } catch {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.gguf';
      input.style.display = 'none';
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) await openModel(file.name);
        input.remove();
      };
      document.body.appendChild(input);
      input.click();
      return;
    }

    if (selected) await openModel(selected);
  }, [openModel]);

  useEffect(() => {
    if (!model || recipe) return;
    const tensors = model.tensors.map(t => ({ name: t.name, currentQuant: t.currentQuant }));
    initRecipe(modelPath ?? 'unknown.gguf', tensors.map(t => t.name), toTargetQuant(tensors[0]?.currentQuant));
  }, [model, modelPath, recipe, initRecipe]);

  const handleTest = useCallback(async () => {
    if (!recipe) return;
    startOperation();
    try {
      const result = await testRecipe(recipe, 512);
      setBenchmarkResult(result);
      setShowResults(true);
      setProfile({ vramEstimate: result.vramAllocatedMb, sizeSavedVsQ8: 0 });
    } catch (e) {
      console.error('Benchmark failed:', e);
    } finally {
      endOperation();
    }
  }, [recipe, startOperation, endOperation, setProfile]);

  const handleSaveRecipe = useCallback(async () => {
    if (!recipe) return;
    try {
      const { save } = await import('@tauri-apps/plugin-dialog');
      const path = await save({ filters: [{ name: 'Recipe JSON', extensions: ['json'] }] });
      if (path && typeof path === 'string') await saveRecipe(path, recipe);
    } catch { /* browser fallback: no-op */ }
  }, [recipe]);

  const handleExport = useCallback(async () => {
    if (!recipe) return;
    try {
      const { save } = await import('@tauri-apps/plugin-dialog');
      const path = await save({ filters: [{ name: 'GGUF', extensions: ['gguf'] }] });
      if (path && typeof path === 'string') await exportGguf(path, recipe);
    } catch { /* browser fallback: no-op */ }
  }, [recipe]);

  const handleLoadRecipe = useCallback(async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({ filters: [{ name: 'Recipe JSON', extensions: ['json'] }] });
      if (selected && typeof selected === 'string') {
        const loaded: RecipeState = await loadRecipe(selected);
        setRecipeState(loaded);
      }
    } catch { /* browser fallback: no-op */ }
  }, [setRecipeState]);

  const selectedTensors = selectedLayerIndex !== null ? getTensorsForLayer(selectedLayerIndex) : [];

  return (
    <div className="h-screen min-h-0 overflow-hidden flex flex-col bg-bg-primary">
      <TitleBar />
      <div className="flex-1 min-h-0 overflow-hidden">
        <AppShell
          toolbar={
            <Toolbar
              modelPath={modelPath}
              hasModel={!!model}
              running={running}
              progress={progress}
              onOpenModel={handleOpenModel}
              onSetAll={assignAll}
              onSaveRecipe={handleSaveRecipe}
              onLoadRecipe={handleLoadRecipe}
              onExport={handleExport}
              onTest={handleTest}
            />
          }
          sidebar={
            <div className="flex flex-col h-full min-h-0">
              <LayerBrowser
                tensors={model?.tensors ?? []}
                selectedLayerIndex={selectedLayerIndex}
                onSelectLayer={setSelectedLayerIndex}
              />
              <BulkAssignPanel onAssign={assignByPattern} disabled={!model || running} />
            </div>
          }
          detail={
            <div className="h-full min-h-0 flex flex-col">
              {modelError && (
                <div className="shrink-0 border-b border-red-500/40 bg-red-950/30 px-4 py-2 text-sm text-red-200">
                  {modelError}
                </div>
              )}
              <div className="flex-1 min-h-0">
                <DetailPanel
                  tensors={selectedTensors}
                  assignments={getAssignments}
                  profile={recipe?.profile ?? null}
                  onAssignQuant={assignQuant}
                />
              </div>
            </div>
          }
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
