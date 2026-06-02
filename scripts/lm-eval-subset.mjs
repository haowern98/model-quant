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

const DEFAULT_EXTRA_PPL_TEXTS = [
  "A quantization workbench should test more than one kind of text. Short technical notes measure whether a model can follow precise vocabulary, while ordinary explanatory prose checks whether common sentence structure remains predictable after tensor conversions.",
  "When a model file is edited, the tensor data and the metadata must continue to agree. A reader should be able to inspect the file, reconstruct the shape of each tensor, and verify that the stored type matches the recipe that produced it.",
  "A software release process often starts with a small local smoke test. The smoke test is not meant to prove every behavior, but it should catch obvious failures before a slower benchmark or a manual review consumes more time.",
  "In a layered neural network, early tensors may influence broad token representations, while later tensors can affect narrower task decisions. A useful editor should make those structures visible instead of forcing users to reason from a flat list of names.",
  "The laboratory report described a simple experiment with water, heat, and measurement error. The student repeated the procedure several times, recorded the observations in a table, and compared the results against the original hypothesis.",
  "A documentation page should explain the behavior of a command without assuming the reader already knows the implementation. Good examples show the input, the expected output, and the reason a particular option would be selected.",
  "Resource measurements can disagree when they describe different parts of a run. Disk size reflects serialized weights, allocated memory reflects loaded buffers, and peak memory can include temporary workspace used during conversion or evaluation.",
  "A medical training question often includes age, symptoms, medications, and a small number of relevant findings. The model must identify which details are important and avoid treating every phrase as equally diagnostic.",
  "In graph theory, a path connects vertices through a sequence of edges. Some graph problems have efficient algorithms, but others require searching a large space of possibilities and become impractical as the graph grows.",
  "A careful benchmark should report the number of samples. A change of one correct answer has a very different meaning when the test contains twenty examples than when it contains thousands of examples.",
  "The operator adjusted the machine, waited for the pressure to stabilize, and then copied the readings into the maintenance log. The pattern suggested that the sensor was working, but the calibration date still needed to be checked.",
  "When users compare two model recipes, they usually care about tradeoffs. A smaller file is useful only if the quality loss, load time, and runtime behavior are acceptable for the workload they intend to run.",
  "The history lesson described a treaty, the political conditions that preceded it, and the changes that followed. Understanding the passage required connecting dates and causes rather than memorizing a single isolated fact.",
  "A command-line tool can be powerful, but it is easy to lose track of a long list of per-tensor overrides. A graphical table can show the current type, target type, estimated size, and invalid choices in one place.",
  "The physics problem involved a cart moving across a table. After the push ended, friction continued to act opposite the direction of motion, so the cart slowed down until it finally came to rest.",
  "A recipe editor should treat unchanged tensors differently from converted tensors. Reporting both counts helps the user understand whether a run copied most of the model or actually transformed the selected weights.",
  "In a database migration, the safest plan is usually incremental. Each step should be reversible or easy to verify, and the final state should be checked against the schema that the application expects.",
  "Some questions are easy because the answer is a familiar fact. Other questions are hard because every answer choice sounds plausible, and the model must prefer the continuation that best follows the prompt.",
  "A rolling perplexity calculation scores long text in pieces. The context window moves forward through the passage, and only the new target tokens are counted for each window so that tokens are not scored twice.",
  "The user interface should not hide uncertainty. If a recipe gains a few correct answers and loses a few others, the result should show both directions rather than pretending the change is a simple win or loss.",
  "A parser for binary model files must be strict about offsets and sizes. If the reader accepts malformed data silently, later stages may fail in ways that are harder to diagnose than the original format error.",
  "The biology passage explained that organisms respond to environmental pressure over many generations. Traits that improve survival can become more common, while traits that reduce survival may become less common.",
  "A scheduling policy can reduce average waiting time for some jobs while increasing it for others. Evaluating the policy requires looking at the aggregate metric and also the individual cases that changed.",
  "Local model users often choose a quantization level by trial and error. A tool that shows expected memory use and then validates the recipe with the actual runtime can make that process less fragile.",
  "A security review asks what happens when an attacker briefly gains elevated access. Even a short window may be enough to modify files, install persistence, or change settings that survive a reboot.",
  "The weather forecast described a low-pressure system moving through the region overnight. By morning, the sky was cloudy and rain had begun, which matched the expected pattern for that system.",
  "In a compiler, the lexer groups characters into tokens, the parser builds a structure from those tokens, and later stages analyze or transform that structure before producing output.",
  "A good diagnostic report separates evidence from interpretation. Raw counts, margins, and timing numbers should be visible, while labels such as low drift or high drift should be treated as summaries.",
  "The museum catalog described a restored instrument, the materials used to build it, and the evidence that connected it to a particular workshop. The curator noted which claims were certain and which were inferred.",
  "A tensor can be valid for one quantization type and invalid for another. Shape, block size, source type, and tensor role all matter when deciding whether a conversion should be offered to the user.",
  "An engineering notebook is useful because it preserves decisions in the order they were made. Later readers can see which assumptions were tested, which measurements supported the design, and which risks still need attention.",
  "The teacher asked the class to compare two explanations for the same observation. One explanation matched the evidence from the experiment, while the other depended on a detail that had not been measured.",
  "When a file format changes, compatibility checks should fail with specific messages. A vague error may be technically correct, but a precise error tells the user which field or tensor caused the problem.",
  "A language model can answer a multiple choice question by ranking continuations. The model does not need to generate a long explanation; it only needs to assign the highest score to the continuation that matches the correct answer.",
  "The financial report separated recurring revenue from one-time payments. That distinction mattered because a single large transaction could make a month look strong even when the underlying trend had not improved.",
  "A recipe that saves memory can still be a poor choice if it slows down loading or damages important tasks. The best comparison is a table that puts size, speed, and quality next to each other.",
  "In chemistry, a balanced equation preserves the number of atoms for each element. Changing a coefficient can fix one side of the equation while accidentally breaking another, so every element must be checked.",
  "A model editor should preserve the original file unless the user explicitly exports a new version. Testing a recipe in memory is useful because experiments can be discarded without creating extra files.",
  "The network administrator reviewed logs from several machines. Most entries were routine, but one repeated connection attempt from an unexpected address required further investigation.",
  "A small benchmark can be stable enough for local iteration without pretending to replace a full evaluation suite. Its purpose is to catch obvious regressions and guide which recipe deserves deeper testing.",
  "The transit map showed several routes that met at the central station. A passenger could reduce travel time by changing trains there, but only if the transfer schedule left enough margin.",
  "When scoring text with a rolling context window, the evaluator must avoid double counting tokens. Each target token should contribute to the loss once, even though surrounding context may appear in more than one window.",
  "A user may not know whether a tensor belongs to attention, feed-forward layers, embeddings, or normalization. Grouping tensors by role can make bulk operations clearer and reduce accidental edits.",
  "The ecology passage described a food web in which producers, consumers, and decomposers exchanged energy. Removing one species could affect several others because the relationships were connected.",
  "A test result with a positive accuracy delta is not automatically proof of improvement. It may represent a handful of borderline answers where small numerical changes moved the selected choice.",
  "The manufacturing team inspected a batch of components and found that most dimensions were within tolerance. A few parts near the limit required additional checks before the batch could be approved.",
  "A command that rewrites model tensors should validate the requested target types before doing expensive work. Early validation makes failures faster and protects the user from waiting for a doomed conversion.",
  "The literature passage used a quiet scene to reveal a character's priorities. The important evidence was not a single sentence, but the contrast between what the character noticed and what they ignored.",
  "In operating systems, virtual memory separates the address space a program sees from the physical memory installed in the machine. Pages can be moved, loaded, or evicted as execution proceeds.",
  "A local benchmark should be reproducible. If the selected rows, prompt format, and scoring method are fixed, a user can compare two recipes without wondering whether the test itself changed.",
  "The navigation system recalculated the route after a bridge closure. The new path was longer, but it avoided the blocked road and arrived sooner than waiting for the original route to clear.",
  "A quantized model can have smaller tensors but still allocate temporary buffers during evaluation. That is why disk size, working set, and peak allocation are related but not identical measurements.",
  "The mathematics explanation introduced a variable, defined the equation, and then substituted known values. Each step followed from the previous one, which made the final calculation easy to audit.",
  "A preview panel should show enough information for a confident decision without overwhelming the user. Detailed audits can be collapsed by default, while summary numbers remain visible at the top.",
  "The historian compared letters, shipping records, and newspaper notices to reconstruct the event. No single document was complete, but together they formed a more reliable account.",
  "A model may be useful for one workload and weak for another. Recipes that preserve chat behavior may not preserve factual multiple choice scoring, and recipes that score well on small tests still need manual review.",
  "The quality of an eval depends on both the examples and the scoring rule. If either one is unclear, users may overinterpret a small change or compare numbers that were produced by different procedures.",
  "The technician replaced a worn connector and repeated the measurement. The signal became stable, which suggested that the original fault was mechanical rather than a software configuration issue.",
  "A frozen subset should include provenance. Recording the generator, source datasets, selection rule, and formatting notes helps future maintainers understand why the rows look the way they do.",
  "The recipe table is a workspace, not just a report. Users should be able to inspect a tensor, change its target type, test the effect, and return to the same point in the model structure.",
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

export function getPplTextsForPreset(presetName = "default") {
  if (presetName === "quick") {
    return DEFAULT_PPL_TEXTS;
  }
  return [...DEFAULT_PPL_TEXTS, ...DEFAULT_EXTRA_PPL_TEXTS];
}

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
    ppl: getPplTextsForPreset(presetName).map((text) => ({ text })),
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
