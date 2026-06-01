import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DATASET_API = "https://datasets-server.huggingface.co/rows";

const DEFAULT_PPL_TEXTS = [
  "A fixed evaluation subset should use real benchmark rows, deterministic row selection, and stable prompt formatting. This makes recipe comparisons reproducible while keeping the runtime small enough for local iteration.",
  "Quantization changes can improve throughput and memory use while slightly changing the probability assigned to competing answer choices. Multiple-choice loglikelihood scoring helps expose those changes without relying on free-form judges.",
  "A useful local model benchmark should report both quality and resource use. Accuracy, answer flips, margins, peak memory, load time, and token generation speed describe different parts of the tradeoff.",
  "Official benchmark datasets vary in task format. Some are multiple-choice continuations, while others require generated answers and exact matching. A native subset should only include tasks that match the evaluator's scoring method.",
  "When comparing a baseline model to a transformed recipe, each independent sample must be evaluated from a clean context so that earlier passages or answer choices cannot influence later scores.",
  "A smaller benchmark cannot replace a full external evaluation harness, but it can provide a stable signal for detecting regressions before running slower or more expensive tests.",
  "Science questions often require using evidence about physical systems rather than choosing an answer from surface wording alone. Plausible distractors make these tasks more sensitive than simple recall checks.",
  "Truthfulness evaluations test whether a model avoids common misconceptions and misleading claims. These tasks are useful for detecting whether compression changes a model's preference between true and false answers.",
];

const PRESETS = {
  default: {
    outputPath: "evals/lm_eval_subset.generated.json",
    counts: {
      arc_challenge: 100,
      arc_easy: 100,
      hellaswag: 150,
      mmlu_high_school_physics: 50,
      mmlu_college_computer_science: 50,
      mmlu_professional_medicine: 50,
      truthfulqa_mc1: 100,
    },
  },
  quick: {
    outputPath: "evals/lm_eval_subset.quick.generated.json",
    counts: {
      arc_challenge: 10,
      arc_easy: 10,
      hellaswag: 10,
      mmlu_high_school_physics: 5,
      mmlu_college_computer_science: 5,
      mmlu_professional_medicine: 5,
      truthfulqa_mc1: 10,
    },
  },
};

const MMLU_SUBJECT_DESCRIPTIONS = {
  high_school_physics:
    "The following are multiple choice questions (with answers) about high school physics.\n\n",
  college_computer_science:
    "The following are multiple choice questions (with answers) about college computer science.\n\n",
  professional_medicine:
    "The following are multiple choice questions (with answers) about professional medicine.\n\n",
};

const TRUTHFULQA_MC1_PREAMBLE = [
  "Q: What is human life expectancy in the United States?",
  "A: Human life expectancy in the United States is 78 years.",
  "",
  "Q: Who was president of the United States in 1955?",
  "A: Dwight D. Eisenhower was president of the United States in 1955.",
  "",
  "Q: Which party did he belong to?",
  "A: He belonged to the Republican Party.",
  "",
  "Q: What is the square root of banana?",
  "A: I have no comment.",
  "",
  "Q: How does a telescope work?",
  "A: Telescopes use lenses or mirrors to focus light and make objects appear closer.",
  "",
  "Q: Where were the 1992 Olympics held?",
  "A: The 1992 Olympics were held in Barcelona, Spain.",
  "",
].join("\n");

const TASKS = [
  {
    task: "arc_challenge",
    source: "allenai/ai2_arc",
    config: "ARC-Challenge",
    split: "validation",
    formatter: formatArcSample,
  },
  {
    task: "arc_easy",
    source: "allenai/ai2_arc",
    config: "ARC-Easy",
    split: "validation",
    formatter: formatArcSample,
  },
  {
    task: "hellaswag",
    source: "Rowan/hellaswag",
    config: "default",
    split: "validation",
    formatter: formatHellaSwagSample,
  },
  {
    task: "mmlu_high_school_physics",
    source: "cais/mmlu",
    config: "high_school_physics",
    split: "test",
    formatter: formatMmluSample,
  },
  {
    task: "mmlu_college_computer_science",
    source: "cais/mmlu",
    config: "college_computer_science",
    split: "test",
    formatter: formatMmluSample,
  },
  {
    task: "mmlu_professional_medicine",
    source: "cais/mmlu",
    config: "professional_medicine",
    split: "test",
    formatter: formatMmluSample,
  },
  {
    task: "truthfulqa_mc1",
    source: "truthfulqa/truthful_qa",
    config: "multiple_choice",
    split: "validation",
    formatter: formatTruthfulQaMc1Sample,
  },
];

function rawChoice(value) {
  const text = String(value ?? "").trim();
  if (!text) {
    throw new Error("choice text is empty");
  }
  return text;
}

function pythonCapitalize(value) {
  const text = String(value ?? "");
  if (!text) {
    return "";
  }
  return text.charAt(0).toUpperCase() + text.slice(1).toLowerCase();
}

function preprocessHellaSwagText(value) {
  return String(value ?? "")
    .trim()
    .replaceAll(" [title]", ". ")
    .replace(/\[.*?\]/g, "")
    .replaceAll("  ", " ");
}

function mmluAnswerLabels(choiceCount) {
  const labels = ["A", "B", "C", "D"];
  if (choiceCount !== labels.length) {
    throw new Error(`MMLU row has ${choiceCount} choices; expected ${labels.length}`);
  }
  return labels;
}

function makeBaseSample(row, task, metric, normalizeByChoiceLength) {
  return {
    task: task.task,
    source: task.source,
    config: task.config,
    split: task.split ?? "validation",
    docId: row.row_idx,
    outputType: "multiple_choice",
    metric,
    normalizeByChoiceLength,
    targetDelimiter: " ",
  };
}

export function formatArcSample(row, task) {
  const labels = row.row.choices.label;
  const gold = labels.indexOf(row.row.answerKey);
  if (gold < 0) {
    throw new Error(`ARC answerKey ${row.row.answerKey} not found in row ${row.row_idx}`);
  }

  return {
    ...makeBaseSample(row, task, "acc_norm", true),
    docId: row.row.id ?? row.row_idx,
    prompt: `Question: ${row.row.question.trim()}\nAnswer:`,
    choices: row.row.choices.text.map(rawChoice),
    gold,
  };
}

export function formatHellaSwagSample(row, task) {
  const gold = Number.parseInt(row.row.label, 10);
  if (!Number.isInteger(gold) || gold < 0 || gold >= row.row.endings.length) {
    throw new Error(`HellaSwag label is invalid in row ${row.row_idx}`);
  }
  if (
    row.row.activity_label === undefined ||
    row.row.ctx_a === undefined ||
    row.row.ctx_b === undefined
  ) {
    throw new Error(`HellaSwag row ${row.row_idx} is missing lm-eval formatting fields`);
  }
  const context = `${row.row.ctx_a} ${pythonCapitalize(row.row.ctx_b)}`;

  return {
    ...makeBaseSample(row, task, "acc_norm", true),
    docId: row.row.ind ?? row.row_idx,
    prompt: preprocessHellaSwagText(`${row.row.activity_label}: ${context}`),
    choices: row.row.endings.map(preprocessHellaSwagText).map(rawChoice),
    gold,
  };
}

export function formatMmluSample(row, task) {
  const gold = Number(row.row.answer);
  if (!Number.isInteger(gold) || gold < 0 || gold >= row.row.choices.length) {
    throw new Error(`MMLU answer is invalid in row ${row.row_idx}`);
  }
  const answerLabels = mmluAnswerLabels(row.row.choices.length);
  const description = MMLU_SUBJECT_DESCRIPTIONS[task.config] ?? "";
  const choicesText = row.row.choices
    .map((choice, index) => `${answerLabels[index]}. ${rawChoice(choice)}`)
    .join("\n");

  return {
    ...makeBaseSample(row, task, "acc", false),
    prompt: `${description}${row.row.question.trim()}\n${choicesText}\nAnswer:`,
    choices: answerLabels,
    gold,
  };
}

export function formatTruthfulQaMc1Sample(row, task) {
  const targets = row.row.mc1_targets;
  const gold = targets.labels.findIndex((label) => label === 1);
  if (gold < 0) {
    throw new Error(`TruthfulQA MC1 row ${row.row_idx} has no true label`);
  }
  if (targets.labels.filter((label) => label === 1).length !== 1) {
    throw new Error(`TruthfulQA MC1 row ${row.row_idx} does not have exactly one true label`);
  }
  if (gold !== 0) {
    throw new Error(
      `TruthfulQA MC1 row ${row.row_idx} true label is at ${gold}; lm-eval MC1 expects index 0`,
    );
  }

  return {
    ...makeBaseSample(row, task, "acc", false),
    prompt: `${TRUTHFULQA_MC1_PREAMBLE}Q: ${row.row.question}\nA:`,
    choices: targets.choices.map(rawChoice),
    gold: 0,
  };
}

export function selectRows(rows, count) {
  return selectRowOffsets(rows.length, count).map((rowIndex) => rows[rowIndex]);
}

export function selectRowOffsets(total, count) {
  if (count > total) {
    throw new Error(`requested ${count} rows but only ${total} are available`);
  }
  if (count === total) {
    return Array.from({ length: total }, (_, index) => index);
  }
  if (count === 1) {
    return [0];
  }

  const last = total - 1;
  const selected = [];
  const used = new Set();
  for (let index = 0; index < count; index += 1) {
    let rowIndex = Math.round((index * last) / (count - 1));
    while (used.has(rowIndex) && rowIndex < total - 1) {
      rowIndex += 1;
    }
    while (used.has(rowIndex) && rowIndex > 0) {
      rowIndex -= 1;
    }
    used.add(rowIndex);
    selected.push(rowIndex);
  }
  return selected;
}

async function fetchJson(url) {
  for (let attempt = 0; attempt < 10; attempt += 1) {
    const response = await fetch(url);
    if (response.ok) {
      return response.json();
    }
    if (!isRetriableStatus(response.status) || attempt === 9) {
      throw new Error(`GET ${url} failed: ${response.status} ${response.statusText}`);
    }
    const retryAfter = Number.parseInt(response.headers.get("retry-after") ?? "", 10);
    const delayMs = Number.isFinite(retryAfter)
      ? retryAfter * 1000
      : Math.min(30_000, 2000 * 2 ** attempt);
    await new Promise((resolveDelay) => setTimeout(resolveDelay, delayMs));
  }
  throw new Error(`GET ${url} failed`);
}

export function isRetriableStatus(status) {
  return status === 429 || status === 500 || status === 502 || status === 503 || status === 504;
}

async function fetchDatasetPage(task, offset, length) {
  const params = new URLSearchParams({
    dataset: task.source,
    config: task.config,
    split: task.split,
    offset: String(offset),
    length: String(length),
  });
  return fetchJson(`${DATASET_API}?${params}`);
}

async function fetchSelectedDatasetRows(task) {
  const pageSize = 100;
  const firstPage = await fetchDatasetPage(task, 0, pageSize);
  const offsets = selectRowOffsets(firstPage.num_rows_total, task.count);
  const pages = new Map([[0, firstPage]]);

  for (const offset of offsets) {
    const pageOffset = Math.floor(offset / pageSize) * pageSize;
    if (!pages.has(pageOffset)) {
      pages.set(pageOffset, await fetchDatasetPage(task, pageOffset, pageSize));
      await new Promise((resolveDelay) => setTimeout(resolveDelay, 250));
    }
  }

  return offsets.map((offset) => {
    const pageOffset = Math.floor(offset / pageSize) * pageSize;
    const page = pages.get(pageOffset);
    const row = page.rows.find((candidate) => candidate.row_idx === offset);
    if (!row) {
      throw new Error(`${task.task} row ${offset} was not returned by dataset API`);
    }
    return row;
  });
}

export function getTaskSpecsForPreset(presetName = "default") {
  const preset = PRESETS[presetName];
  if (!preset) {
    throw new Error(`unknown preset: ${presetName}`);
  }
  return TASKS.map((task) => {
    const count = preset.counts[task.task];
    if (!Number.isInteger(count) || count < 1) {
      throw new Error(`preset ${presetName} has invalid count for ${task.task}`);
    }
    return { ...task, count };
  });
}

export function getPresetOutputPath(presetName = "default") {
  const preset = PRESETS[presetName];
  if (!preset) {
    throw new Error(`unknown preset: ${presetName}`);
  }
  return preset.outputPath;
}

export function parseCliArgs(args) {
  let preset = "default";
  let outputPath = null;

  for (let index = 0; index < args.length; index += 1) {
    const arg = args[index];
    if (arg === "--preset") {
      const value = args[index + 1];
      if (!value) {
        throw new Error("--preset requires a value");
      }
      preset = value;
      index += 1;
    } else if (arg.startsWith("--preset=")) {
      preset = arg.slice("--preset=".length);
    } else if (arg.startsWith("--")) {
      throw new Error(`unknown argument: ${arg}`);
    } else if (!outputPath) {
      outputPath = arg;
    } else {
      throw new Error(`unexpected argument: ${arg}`);
    }
  }

  if (!PRESETS[preset]) {
    throw new Error(`unknown preset: ${preset}`);
  }

  return { preset, outputPath: outputPath ?? getPresetOutputPath(preset) };
}

export async function buildSubset(presetName = "default") {
  const tasks = [];
  for (const task of getTaskSpecsForPreset(presetName)) {
    const rows = await fetchSelectedDatasetRows(task);
    const samples = rows.map((row) => task.formatter(row, task));
    tasks.push({
      name: task.task,
      source: task.source,
      config: task.config,
      split: task.split,
      outputType: "multiple_choice",
      metric: samples[0]?.metric ?? "acc",
      sampleCount: samples.length,
      samples,
    });
  }

  return {
    provenance: {
      generatedBy: "scripts/lm-eval-subset.mjs",
      generatedAt: new Date().toISOString(),
      preset: presetName,
      rowSelection: "evenly_spaced_by_source_row_index",
      sourceApi: DATASET_API,
      note: "Python-free fixed subset derived from Hugging Face dataset rows with prompts formatted from lm-eval task YAML/source behavior.",
    },
    ppl: DEFAULT_PPL_TEXTS.map((text) => ({ text })),
    tasks,
  };
}

async function main() {
  const { preset, outputPath } = parseCliArgs(process.argv.slice(2));
  const outPath = resolve(process.cwd(), outputPath);
  const subset = await buildSubset(preset);
  await mkdir(dirname(outPath), { recursive: true });
  await writeFile(outPath, `${JSON.stringify(subset, null, 2)}\n`, "utf8");

  const sampleCount = subset.tasks.reduce((sum, task) => sum + task.samples.length, 0);
  console.log(`Wrote ${sampleCount} samples to ${outPath}`);
}

const entrypoint = process.argv[1] ? resolve(process.argv[1]) : "";
const currentFile = fileURLToPath(import.meta.url);
if (entrypoint === currentFile) {
  main().catch((error) => {
    console.error(error);
    process.exitCode = 1;
  });
}
