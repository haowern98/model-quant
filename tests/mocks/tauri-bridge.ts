import type { ModelInfo, RecipeState, QuantType, AssignPattern } from '../../src/types';

export function createMockBridge() {
  const mockModel: ModelInfo = {
    metadata: {
      name: 'Mock-Llama-3-8B',
      architecture: 'llama',
      totalParams: 8_030_000_000,
      totalSizeFp16: 14.96 * 1024 * 1024 * 1024,
    },
    tensors: [
      { name: 'tok_embeddings.weight', shape: [128256, 4096], currentQuant: 'Q4_K_M', sizeBytes: 262_000_000, layerIndex: -1, layerGroup: 'embedding' },
      { name: 'layers.0.attention.wq.weight', shape: [4096, 4096], currentQuant: 'Q4_K_M', sizeBytes: 80_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'layers.0.attention.wk.weight', shape: [4096, 1024], currentQuant: 'Q4_K_M', sizeBytes: 20_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'layers.0.attention.wv.weight', shape: [4096, 1024], currentQuant: 'Q4_K_M', sizeBytes: 20_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'layers.0.attention.wo.weight', shape: [4096, 4096], currentQuant: 'Q4_K_M', sizeBytes: 80_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'layers.0.feed_forward.w1.weight', shape: [14336, 4096], currentQuant: 'Q4_K_M', sizeBytes: 117_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'layers.0.feed_forward.w2.weight', shape: [4096, 14336], currentQuant: 'Q4_K_M', sizeBytes: 117_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'layers.0.feed_forward.w3.weight', shape: [14336, 4096], currentQuant: 'Q4_K_M', sizeBytes: 117_000_000, layerIndex: 0, layerGroup: 'attention' },
      { name: 'output_norm.weight', shape: [4096], currentQuant: 'Q4_K_M', sizeBytes: 16_000, layerIndex: -1, layerGroup: 'output_norm' },
      { name: 'output.weight', shape: [128256, 4096], currentQuant: 'Q4_K_M', sizeBytes: 262_000_000, layerIndex: -1, layerGroup: 'output' },
    ],
    currentUniformQuant: 'Q4_K_M',
    totalSizeBytes: 4_920_000_000,
  };

  let recipe: RecipeState = {
    id: 'mock-recipe',
    baseModel: 'mock-model.gguf',
    assignments: mockModel.tensors.map(t => ({ tensorName: t.name, quantType: t.currentQuant })),
    profile: null,
    status: 'draft',
  };

  const handlers: Record<string, (args?: Record<string, unknown>) => unknown> = {
    open_model: (_args) => mockModel,
    get_tensors: () => mockModel.tensors,
    assign_quant: (args) => {
      const { tensorNames, quantType } = args!;
      const names = tensorNames as string[];
      recipe.assignments = recipe.assignments.map(a =>
        names.includes(a.tensorName) ? { ...a, quantType: quantType as QuantType } : a
      );
      recipe.status = 'draft';
      return recipe;
    },
    assign_all: (args) => {
      const { quantType } = args!;
      recipe.assignments = recipe.assignments.map(a => ({ ...a, quantType: quantType as QuantType }));
      recipe.status = 'draft';
      return recipe;
    },
    assign_by_pattern: (args) => {
      const { pattern, quantType } = args!;
      const isMatch = (name: string, p: AssignPattern): boolean => {
        if (p === 'all_attn') return name.includes('attention');
        if (p === 'all_ffn') return name.includes('feed_forward');
        if (p === 'all_embeddings') return name.includes('embedding') || name.includes('output');
        return true;
      };
      recipe.assignments = recipe.assignments.map(a =>
        isMatch(a.tensorName, pattern as AssignPattern) ? { ...a, quantType: quantType as QuantType } : a
      );
      recipe.status = 'draft';
      return recipe;
    },
    test_recipe: (_args) => ({
      promptEvalTps: 1247,
      tokenGenTps: 88.3,
      ttftMs: 412,
      vramPeakMb: 5820,
      vramAllocatedMb: 5760,
      diskSizeMb: 5780,
      elapsedMs: 3200,
    }),
    save_recipe: () => { recipe.status = 'saved'; },
    export_gguf: () => {},
    load_recipe: () => recipe,
    list_recipes: () => [{ id: 'mock-recipe', baseModel: 'mock-model.gguf', status: 'saved', createdAt: '2026-05-23' }],
  };

  return async (cmd: string, args?: Record<string, unknown>): Promise<unknown> => {
    const handler = handlers[cmd];
    if (!handler) throw new Error(`Unknown command: ${cmd}`);
    return handler(args);
  };
}
