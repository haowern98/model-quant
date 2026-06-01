import { useState, useCallback } from 'react';
import type { ModelInfo, TensorInfo } from '../types';
import { openModel as openModelCmd } from '../lib/tauri-bridge';

interface ModelState {
  model: ModelInfo | null;
  modelPath: string | null;
  loading: boolean;
  error: string | null;
}

export function useModel() {
  const [state, setState] = useState<ModelState>({
    model: null, modelPath: null, loading: false, error: null,
  });

  const openModel = useCallback(async (path: string): Promise<ModelInfo | null> => {
    setState(s => ({ ...s, loading: true, error: null }));
    try {
      const model = await openModelCmd(path);
      setState({ model, modelPath: path, loading: false, error: null });
      return model;
    } catch (e) {
      setState(s => ({ ...s, loading: false, error: (e as Error).message }));
      return null;
    }
  }, []);

  const getTensorsForLayer = useCallback((layerIndex: number): TensorInfo[] => {
    if (!state.model) return [];
    return state.model.tensors.filter(t => t.layerIndex === layerIndex);
  }, [state.model]);

  return { ...state, openModel, getTensorsForLayer };
}
