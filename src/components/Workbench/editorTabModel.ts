export const EVAL_RESULTS_TAB_ID = "eval-results";
export const GPQA_DETAILS_TAB_ID = "benchmark:gpqa_diamond:details";
export const GPQA_DATASET_TAB_ID = "benchmark:gpqa_diamond:dataset";

export type EditorTab =
  | {
      id: `layer:${number}`;
      kind: "layer";
      layerIndex: number;
    }
  | {
      id: typeof EVAL_RESULTS_TAB_ID;
      kind: "eval-results";
    }
  | {
      id: typeof GPQA_DETAILS_TAB_ID;
      kind: "gpqa-details";
    }
  | {
      id: typeof GPQA_DATASET_TAB_ID;
      kind: "gpqa-dataset";
    };

export function layerEditorTab(layerIndex: number): EditorTab {
  return {
    id: `layer:${layerIndex}`,
    kind: "layer",
    layerIndex,
  };
}

export function gpqaDetailsEditorTab(): EditorTab {
  return {
    id: GPQA_DETAILS_TAB_ID,
    kind: "gpqa-details",
  };
}

export function gpqaDatasetEditorTab(): EditorTab {
  return {
    id: GPQA_DATASET_TAB_ID,
    kind: "gpqa-dataset",
  };
}

export function editorTabLabel(tab: EditorTab): string {
  if (tab.kind === "eval-results") return "Eval Results";
  if (tab.kind === "gpqa-details") return "GPQA Diamond Details";
  if (tab.kind === "gpqa-dataset") return "GPQA Diamond Dataset";
  if (tab.layerIndex < 0) return "Global tensors";
  return `Layer ${tab.layerIndex}`;
}
