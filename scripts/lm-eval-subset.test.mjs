import test from "node:test";
import assert from "node:assert/strict";

import {
  formatArcSample,
  formatHellaSwagSample,
  formatMmluSample,
  formatTruthfulQaMc1Sample,
  isRetriableStatus,
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
  assert.deepEqual(sample.choices, [" paper", " iron", " glass"]);
  assert.equal(sample.gold, 1);
  assert.equal(sample.normalizeByChoiceLength, true);
});

test("formats HellaSwag rows as acc_norm continuation samples", () => {
  const sample = formatHellaSwagSample(
    {
      row_idx: 3,
      row: {
        ind: 24,
        ctx: "A man is sitting on a roof. he",
        endings: ["falls asleep.", "starts pulling up roofing."],
        label: "1",
      },
    },
    { task: "hellaswag", source: "Rowan/hellaswag", config: "default" },
  );

  assert.equal(sample.task, "hellaswag");
  assert.equal(sample.docId, 24);
  assert.equal(sample.prompt, "A man is sitting on a roof. he");
  assert.deepEqual(sample.choices, [" falls asleep.", " starts pulling up roofing."]);
  assert.equal(sample.gold, 1);
  assert.equal(sample.normalizeByChoiceLength, true);
});

test("formats MMLU rows as single-gold multiple-choice samples", () => {
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
  assert.equal(sample.prompt, "Question: What is 2 + 2?\nAnswer:");
  assert.deepEqual(sample.choices, [" 3", " 4", " 5", " 6"]);
  assert.equal(sample.gold, 1);
  assert.equal(sample.normalizeByChoiceLength, false);
});

test("formats TruthfulQA MC1 rows with the sole true label as gold", () => {
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
  assert.equal(sample.prompt, "Question: Can a coin remember previous flips?\nAnswer:");
  assert.deepEqual(sample.choices, [
    " No, each fair flip is independent.",
    " Yes, coins remember.",
  ]);
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
