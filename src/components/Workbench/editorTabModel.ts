export const EVAL_RESULTS_TAB_ID = "eval-results";

export type EditorTab =
  | {
      id: `layer:${number}`;
      kind: "layer";
      layerIndex: number;
    }
  | {
      id: typeof EVAL_RESULTS_TAB_ID;
      kind: "eval-results";
    };

export function layerEditorTab(layerIndex: number): EditorTab {
  return {
    id: `layer:${layerIndex}`,
    kind: "layer",
    layerIndex,
  };
}

export function editorTabLabel(tab: EditorTab): string {
  if (tab.kind === "eval-results") return "Eval Results";
  if (tab.layerIndex < 0) return "Global tensors";
  return `Layer ${tab.layerIndex}`;
}
