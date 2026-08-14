export type ModelLoadConfig = {
  seed: string;
  thinking: "on" | "off";
  temperature: string;
  topK: string;
  repeatPenalty: string;
  presencePenalty: string;
  topP: string;
  minP: string;
  contextWindow: string;
};

export const DEFAULT_MODEL_LOAD_CONFIG: ModelLoadConfig = {
  seed: "",
  thinking: "off",
  temperature: "0",
  topK: "40",
  repeatPenalty: "1.1",
  presencePenalty: "0",
  topP: "0.95",
  minP: "0.05",
  contextWindow: "20000",
};

export type ChatMessageData = {
  id: string;
  role: "user" | "assistant";
  content: string;
  reasoning?: string;
  model?: string;
  tokensPerSecond?: number;
  promptTokens?: number;
  durationSeconds?: number;
  finishReason?: string;
  seed?: number;
  trace?: ChatTraceReference;
};

export type ChatTraceReference = {
  conversationId: string;
  assistantMessageId: string;
  modelFingerprint: string;
  tokenCount: number;
  status?: "saving" | "saved" | "failed";
};

export type ChatConversation = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: ChatMessageData[];
};

export type ChatConversationSummary = Pick<ChatConversation, "id" | "title" | "updatedAt">;

export type ChatGenerationConfig = {
  seed?: number;
  thinking: boolean;
  temperature: number;
  topK: number;
  repeatPenalty: number;
  presencePenalty: number;
  topP: number;
  minP: number;
  contextWindow: number;
};

export function chatConfigFromModelLoadConfig(
  config: ModelLoadConfig,
): ChatGenerationConfig | string {
  const seed = optionalInteger(config.seed, 0, 4_294_967_294, "Seed");
  if (typeof seed === "string") return seed;
  const temperature = numberInRange(config.temperature, 0, 2, "Temperature");
  if (typeof temperature === "string") return temperature;
  const topK = integerInRange(config.topK, 0, 1000, "Top K Sampling");
  if (typeof topK === "string") return topK;
  const repeatPenalty = numberInRange(config.repeatPenalty, 0, 3, "Repeat Penalty");
  if (typeof repeatPenalty === "string") return repeatPenalty;
  const presencePenalty = numberInRange(config.presencePenalty, -2, 2, "Presence Penalty");
  if (typeof presencePenalty === "string") return presencePenalty;
  const topP = numberInRange(config.topP, 0, 1, "Top P Sampling");
  if (typeof topP === "string") return topP;
  const minP = numberInRange(config.minP, 0, 1, "Min P Sampling");
  if (typeof minP === "string") return minP;
  const contextWindow = integerInRange(config.contextWindow, 1, Number.MAX_SAFE_INTEGER, "Context Window");
  if (typeof contextWindow === "string") return contextWindow;

  return {
    seed,
    thinking: config.thinking === "on",
    temperature,
    topK,
    repeatPenalty,
    presencePenalty,
    topP,
    minP,
    contextWindow,
  };
}

function optionalInteger(value: string, min: number, max: number, label: string): number | undefined | string {
  if (value.trim() === "") return undefined;
  return integerInRange(value, min, max, label);
}

function integerInRange(value: string, min: number, max: number, label: string): number | string {
  if (!/^\d+$/.test(value.trim())) return `${label} must be a whole number.`;
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < min || parsed > max) {
    return `${label} must be between ${min} and ${max}.`;
  }
  return parsed;
}

function numberInRange(value: string, min: number, max: number, label: string): number | string {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed < min || parsed > max) {
    return `${label} must be between ${min} and ${max}.`;
  }
  return parsed;
}
