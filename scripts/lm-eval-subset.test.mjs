import test from "node:test";
import assert from "node:assert/strict";

import {
  formatArcSample,
  formatHellaSwagSample,
  formatMmluSample,
  formatTruthfulQaMc1Sample,
  getPresetOutputPath,
  getPplTextsForPreset,
  getTaskSpecsForPreset,
  isRetriableStatus,
  parseCliArgs,
  selectRowOffsets,
  selectRows,
} from "./lm-eval-subset.mjs";

test("formats ARC rows as acc_norm multiple-choice samples", () => {
  const sample = formatArcSample(
    {
      row_idx: 7,
      row: {
        id: "Mercury_1",
        question: "Which object is attracted by a magnet?",
        choices: { text: ["paper", "iron", "glass"], label: ["A", "B", "C"] },
        answerKey: "B",
      },
    },
    { task: "arc_challenge", source: "allenai/ai2_arc", config: "ARC-Challenge" },
  );

  assert.equal(sample.task, "arc_challenge");
  assert.equal(sample.docId, "Mercury_1");
  assert.equal(sample.metric, "acc_norm");
  assert.equal(sample.prompt, "Question: Which object is attracted by a magnet?\nAnswer:");
  assert.equal(sample.targetDelimiter, " ");
  assert.deepEqual(sample.choices, ["paper", "iron", "glass"]);
  assert.equal(sample.gold, 1);
  assert.equal(sample.normalizeByChoiceLength, true);
});

test("formats HellaSwag rows as acc_norm continuation samples", () => {
  const sample = formatHellaSwagSample(
    {
      row_idx: 3,
      row: {
        ind: 24,
        activity_label: "Roof repair",
        ctx_a: "A man is sitting on a roof.",
        ctx_b: "he pulls up old shingles.",
        endings: [" [title] Falls asleep.", "starts pulling up roofing."],
        label: "1",
      },
    },
    { task: "hellaswag", source: "Rowan/hellaswag", config: "default" },
  );

  assert.equal(sample.task, "hellaswag");
  assert.equal(sample.docId, 24);
  assert.equal(sample.prompt, "Roof repair: A man is sitting on a roof. He pulls up old shingles.");
  assert.equal(sample.targetDelimiter, " ");
  assert.deepEqual(sample.choices, ["Falls asleep.", "starts pulling up roofing."]);
  assert.equal(sample.gold, 1);
  assert.equal(sample.normalizeByChoiceLength, true);
});

test("formats MMLU rows with default lm-eval answer-letter continuations", () => {
  const sample = formatMmluSample(
    {
      row_idx: 2,
      row: {
        question: "What is 2 + 2?",
        choices: ["3", "4", "5", "6"],
        answer: 1,
      },
    },
    {
      task: "mmlu_high_school_physics",
      source: "cais/mmlu",
      config: "high_school_physics",
    },
  );

  assert.equal(sample.task, "mmlu_high_school_physics");
  assert.equal(sample.metric, "acc");
  assert.equal(
    sample.prompt,
    [
      "The following are multiple choice questions (with answers) about high school physics.",
      "",
      "What is 2 + 2?",
      "A. 3",
      "B. 4",
      "C. 5",
      "D. 6",
      "Answer:",
    ].join("\n"),
  );
  assert.equal(sample.targetDelimiter, " ");
  assert.deepEqual(sample.choices, ["A", "B", "C", "D"]);
  assert.equal(sample.gold, 1);
  assert.equal(sample.normalizeByChoiceLength, false);
});

test("formats TruthfulQA MC1 rows with the lm-eval MC1 preamble", () => {
  const sample = formatTruthfulQaMc1Sample(
    {
      row_idx: 5,
      row: {
        question: "Can a coin remember previous flips?",
        mc1_targets: {
          choices: ["No, each fair flip is independent.", "Yes, coins remember."],
          labels: [1, 0],
        },
      },
    },
    { task: "truthfulqa_mc1", source: "truthfulqa/truthful_qa", config: "multiple_choice" },
  );

  assert.equal(sample.task, "truthfulqa_mc1");
  assert.equal(sample.metric, "acc");
  assert.match(sample.prompt, /^Q: What is human life expectancy in the United States\?/);
  assert.match(sample.prompt, /\nQ: Can a coin remember previous flips\?\nA:$/);
  assert.equal(sample.targetDelimiter, " ");
  assert.deepEqual(sample.choices, ["No, each fair flip is independent.", "Yes, coins remember."]);
  assert.equal(sample.gold, 0);
  assert.equal(sample.normalizeByChoiceLength, false);
});

test("selectRows uses deterministic evenly-spaced row indices", () => {
  const rows = Array.from({ length: 10 }, (_, row_idx) => ({ row_idx, row: { value: row_idx } }));

  assert.deepEqual(
    selectRows(rows, 4).map((row) => row.row_idx),
    [0, 3, 6, 9],
  );
});

test("selectRowOffsets selects deterministic offsets without fetching all rows", () => {
  assert.deepEqual(selectRowOffsets(10, 4), [0, 3, 6, 9]);
  assert.deepEqual(selectRowOffsets(299, 5), [0, 75, 149, 224, 298]);
});

test("isRetriableStatus covers rate limits and transient server errors", () => {
  assert.equal(isRetriableStatus(429), true);
  assert.equal(isRetriableStatus(500), true);
  assert.equal(isRetriableStatus(502), true);
  assert.equal(isRetriableStatus(503), true);
  assert.equal(isRetriableStatus(504), true);
  assert.equal(isRetriableStatus(404), false);
});

test("default preset keeps the frozen generated subset shape", () => {
  const counts = Object.fromEntries(
    getTaskSpecsForPreset("default").map((task) => [task.task, task.count]),
  );

  assert.equal(getPresetOutputPath("default"), "evals/lm_eval_subset.generated.json");
  assert.deepEqual(counts, {
    arc_challenge: 70,
    arc_easy: 70,
    hellaswag: 70,
    mmlu_high_school_physics: 35,
    mmlu_college_computer_science: 35,
    mmlu_professional_medicine: 35,
    truthfulqa_mc1: 70,
  });
});

test("quick preset writes a separate smaller official-row subset", () => {
  const counts = Object.fromEntries(
    getTaskSpecsForPreset("quick").map((task) => [task.task, task.count]),
  );

  assert.equal(getPresetOutputPath("quick"), "evals/lm_eval_subset.quick.generated.json");
  assert.deepEqual(counts, {
    arc_challenge: 10,
    arc_easy: 10,
    hellaswag: 10,
    mmlu_high_school_physics: 5,
    mmlu_college_computer_science: 5,
    mmlu_professional_medicine: 5,
    truthfulqa_mc1: 10,
  });
});

test("default preset uses a larger rolling-PPL corpus than quick", () => {
  assert.ok(getPplTextsForPreset("default").length > getPplTextsForPreset("quick").length);
  assert.equal(getPplTextsForPreset("quick").length, 8);
});

test("CLI defaults to default output and can generate quick independently", () => {
  assert.deepEqual(parseCliArgs([]), {
    preset: "default",
    outputPath: "evals/lm_eval_subset.generated.json",
  });
  assert.deepEqual(parseCliArgs(["--preset", "quick"]), {
    preset: "quick",
    outputPath: "evals/lm_eval_subset.quick.generated.json",
  });
  assert.deepEqual(parseCliArgs(["--preset=quick", "tmp/quick.json"]), {
    preset: "quick",
    outputPath: "tmp/quick.json",
  });
});
