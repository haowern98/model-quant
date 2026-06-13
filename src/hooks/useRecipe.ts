import { useState, useCallback, useMemo } from 'react';
import type { RecipeState, QuantType, AssignPattern } from '../types';
import { assignQuant as assignQuantCmd, assignAll as assignAllCmd, assignByPattern as assignByPatternCmd } from '../lib/tauri-bridge';

export function useRecipe() {
  const [recipe, setRecipe] = useState<RecipeState | null>(null);

  const resetRecipeForModel = useCallback((modelPath: string, tensors: { name: string; currentQuant: string }[]) => {
    setRecipe({
      id: `${Date.now()}`,
      baseModel: modelPath,
      assignments: tensors.map(t => ({
        tensorName: t.name,
        quantType: t.currentQuant,
        sourceQuant: t.currentQuant,
      })),
      profile: null,
      status: 'draft',
    });
  }, []);

  const setRecipeState = useCallback((nextRecipe: RecipeState) => {
    setRecipe(nextRecipe);
  }, []);

  const assignQuant = useCallback(async (tensorName: string, quantType: QuantType) => {
    if (!recipe) return;
    const updated = await assignQuantCmd([tensorName], quantType);
    setRecipe(updated);
  }, [recipe]);

  const assignAll = useCallback(async (quantType: QuantType) => {
    if (!recipe) return;
    const updated = await assignAllCmd(quantType);
    setRecipe(updated);
  }, [recipe]);

  const assignByPattern = useCallback(async (pattern: AssignPattern, quantType: QuantType) => {
    if (!recipe) return;
    const updated = await assignByPatternCmd(pattern, quantType);
    setRecipe(updated);
  }, [recipe]);

  const setProfile = useCallback((profile: RecipeState['profile']) => {
    setRecipe(r => r ? { ...r, profile, status: 'profiled' as const } : null);
  }, []);

  const getAssignments = useMemo((): Record<string, QuantType> => {
    if (!recipe) return {};
    const map: Record<string, QuantType> = {};
    for (const a of recipe.assignments) {
      if (a.sourceQuant && a.quantType !== a.sourceQuant) {
        map[a.tensorName] = a.quantType;
      }
    }
    return map;
  }, [recipe]);

  return { recipe, resetRecipeForModel, setRecipeState, assignQuant, assignAll, assignByPattern, setProfile, getAssignments };
}
