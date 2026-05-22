import { useCallback, useState } from 'react';
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
import { testRecipe, saveRecipe, loadRecipe, exportGguf } from './lib/tauri-bridge';
import type { BenchmarkResult } from './types';
import { setMockInvoke } from './lib/tauri-bridge';
import { createMockBridge } from '../tests/mocks/tauri-bridge';

// Auto-inject mock bridge when running in browser without Tauri
if (typeof window !== 'undefined') {
  try {
    setMockInvoke(createMockBridge());
  } catch { /* mock already set */ }
}

function App() {
  const { model, modelPath, openModel, getTensorsForLayer } = useModel();
  const { recipe, initRecipe, assignQuant, assignAll, assignByPattern, setProfile, getAssignments } = useRecipe();
  const { progress, running, startOperation, endOperation } = useProgress();

  const [selectedLayerIndex, setSelectedLayerIndex] = useState<number | null>(null);
  const [benchmarkResult, setBenchmarkResult] = useState<BenchmarkResult | null>(null);
  const [showResults, setShowResults] = useState(false);

  const handleOpenModel = useCallback(async () => {
    try {
      const { open } = await import('@tauri-apps/plugin-dialog');
      const selected = await open({
        filters: [{ name: 'GGUF', extensions: ['gguf'] }],
      });
      if (selected && typeof selected === 'string') {
        await openModel(selected);
      }
    } catch {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.gguf';
      input.onchange = async (e) => {
        const file = (e.target as HTMLInputElement).files?.[0];
        if (file) await openModel(file.name);
      };
      input.click();
    }
  }, [openModel]);

  // When model loads, init recipe
  if (model && !recipe) {
    const tensors = model.tensors.map(t => ({ name: t.name, currentQuant: t.currentQuant }));
    initRecipe(modelPath ?? 'unknown.gguf', tensors.map(t => t.name), tensors[0]?.currentQuant ?? 'Q4_K_M');
  }

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
        const loaded = await loadRecipe(selected);
        initRecipe(loaded.baseModel, loaded.assignments.map(a => a.tensorName), loaded.assignments[0]?.quantType ?? 'Q4_K_M');
      }
    } catch { /* browser fallback: no-op */ }
  }, [initRecipe]);

  const selectedTensors = selectedLayerIndex !== null ? getTensorsForLayer(selectedLayerIndex) : [];

  return (
    <div className="h-screen flex flex-col bg-bg-primary">
      <TitleBar />
      <div className="flex-1">
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
            <div className="flex flex-col h-full">
              <LayerBrowser
                tensors={model?.tensors ?? []}
                selectedLayerIndex={selectedLayerIndex}
                onSelectLayer={setSelectedLayerIndex}
              />
              <BulkAssignPanel onAssign={assignByPattern} disabled={!model || running} />
            </div>
          }
          detail={
            <DetailPanel
              tensors={selectedTensors}
              assignments={getAssignments}
              profile={recipe?.profile ?? null}
              onAssignQuant={assignQuant}
            />
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
