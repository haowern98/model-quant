import { listen } from "@tauri-apps/api/event";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  generateChatResponse,
  generateChatTitle,
  listChatConversations,
  loadChatModel,
  loadChatConversation,
  saveChatConversation,
  unloadChatModel,
} from "../../../lib/tauri-bridge";
import {
  type ChatConversation,
  type ChatConversationSummary,
  type ChatMessageData,
  type ModelLoadConfig,
  chatConfigFromModelLoadConfig,
} from "./chatTypes";

type ChatStreamDelta = {
  conversationId: string;
  content: string;
  reasoning: string;
};

export function useChatSession(modelConfig: ModelLoadConfig) {
  const [conversations, setConversations] = useState<Record<string, ChatConversation>>({});
  const [summaries, setSummaries] = useState<ChatConversationSummary[]>([]);
  const [sendingConversationId, setSendingConversationId] = useState<string | null>(null);
  const [modelLoading, setModelLoading] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const conversationsRef = useRef(conversations);

  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);

  useEffect(() => {
    void listChatConversations().then(setSummaries).catch(() => undefined);
  }, []);

  const createConversation = useCallback((): ChatConversation => {
    const now = new Date().toISOString();
    const conversation = {
      id: chatId(),
      title: "New chat",
      createdAt: now,
      updatedAt: now,
      messages: [],
    };
    setConversations((current) => ({ ...current, [conversation.id]: conversation }));
    return conversation;
  }, []);

  const openConversation = useCallback(async (id: string) => {
    const current = conversationsRef.current[id];
    if (current) return current;
    const conversation = await loadChatConversation(id);
    setConversations((existing) => ({ ...existing, [id]: conversation }));
    return conversation;
  }, []);

  const loadModel = useCallback(async () => {
    const config = chatConfigFromModelLoadConfig(modelConfig);
    if (typeof config === "string") {
      setError(config);
      return;
    }
    setError(null);
    setModelLoading(true);
    try {
      await loadChatModel(config);
      setModelLoaded(true);
    } catch (loadError) {
      setModelLoaded(false);
      setError(errorMessage(loadError));
    } finally {
      setModelLoading(false);
    }
  }, [modelConfig]);

  const unloadModel = useCallback(async () => {
    setError(null);
    setModelLoading(true);
    try {
      await unloadChatModel();
      setModelLoaded(false);
    } catch (unloadError) {
      setError(errorMessage(unloadError));
    } finally {
      setModelLoading(false);
    }
  }, []);

  const sendMessage = useCallback(async (conversationId: string, text: string, traceEnabled = false) => {
    const conversation = conversationsRef.current[conversationId];
    const content = text.trim();
    if (!conversation || !content || sendingConversationId || !modelLoaded) return;

    setError(null);
    setSendingConversationId(conversationId);
    const userMessage: ChatMessageData = { id: messageId(), role: "user", content };
    const assistantMessage: ChatMessageData = { id: messageId(), role: "assistant", content: "" };
    const pendingConversation = {
      ...conversation,
      updatedAt: new Date().toISOString(),
      messages: [...conversation.messages, userMessage, assistantMessage],
    };
    setConversations((current) => ({ ...current, [conversationId]: pendingConversation }));
    const requestMessages = pendingConversation.messages
      .filter((message) => message.id !== assistantMessage.id)
      .map(({ role, content: messageContent, reasoning }) => ({ role, content: messageContent, reasoning }));

    let unlisten: (() => void) | undefined;
    try {
      unlisten = await listen<ChatStreamDelta>("chat-stream-delta", (event) => {
        const delta = event.payload;
        if (delta.conversationId !== conversationId) return;
        setConversations((current) => {
          const active = current[conversationId];
          if (!active) return current;
          return {
            ...current,
            [conversationId]: {
              ...active,
              messages: active.messages.map((message) =>
                message.id === assistantMessage.id
                  ? {
                      ...message,
                      content: message.content + delta.content,
                      reasoning: `${message.reasoning ?? ""}${delta.reasoning}` || undefined,
                    }
                  : message,
              ),
            },
          };
        });
      });
      const response = await generateChatResponse({
        conversationId,
        messages: requestMessages,
        traceEnabled,
        assistantMessageId: traceEnabled ? assistantMessage.id : undefined,
      });
      const completedConversation = await finishConversation(
        assistantMessage.id,
        response,
        conversationsRef.current[conversationId] ?? pendingConversation,
      );
      setConversations((current) => ({ ...current, [conversationId]: completedConversation }));
      try {
        const title = completedConversation.title === "New chat"
          ? await titleForConversation(conversationId, completedConversation)
          : completedConversation.title;
        const savedConversation = { ...completedConversation, title, updatedAt: new Date().toISOString() };
        await saveChatConversation(savedConversation);
        setConversations((current) => ({ ...current, [conversationId]: savedConversation }));
        setSummaries((current) => sortSummaries([
          ...current.filter((summary) => summary.id !== conversationId),
          { id: conversationId, title, updatedAt: savedConversation.updatedAt },
        ]));
      } catch (persistenceError) {
        setError(errorMessage(persistenceError));
      }
    } catch (generationError) {
      setError(errorMessage(generationError));
      setConversations((current) => ({
        ...current,
        [conversationId]: {
          ...pendingConversation,
          messages: pendingConversation.messages.filter((message) => message.id !== assistantMessage.id),
        },
      }));
    } finally {
      unlisten?.();
      setSendingConversationId(null);
    }
  }, [modelLoaded, sendingConversationId]);

  return {
    conversations,
    summaries,
    sendingConversationId,
    modelLoading,
    modelLoaded,
    error,
    createConversation,
    openConversation,
    loadModel,
    unloadModel,
    sendMessage,
  };
}

async function finishConversation(
  assistantId: string,
  response: Awaited<ReturnType<typeof generateChatResponse>>,
  conversation: ChatConversation,
): Promise<ChatConversation> {
  const parsed = response.reasoning ? { content: response.content, reasoning: response.reasoning } : splitThinking(response.content);
  const assistant = {
    id: assistantId,
    role: "assistant" as const,
    content: parsed.content,
    reasoning: parsed.reasoning || undefined,
    model: response.model,
    tokensPerSecond: response.tokensPerSecond,
    promptTokens: response.promptTokens,
    durationSeconds: response.durationSeconds,
    finishReason: response.finishReason,
    seed: response.seed,
    trace: response.trace,
  };
  return {
    ...conversation,
    updatedAt: new Date().toISOString(),
    messages: conversation.messages.map((message) => message.id === assistantId ? assistant : message),
  };
}

async function titleForConversation(
  conversationId: string,
  conversation: ChatConversation,
): Promise<string> {
  try {
    const title = await generateChatTitle({
      conversationId,
      messages: conversation.messages.map(({ role, content, reasoning }) => ({ role, content, reasoning })),
    });
    if (title.trim()) return title;
  } catch {
    // ponytail: title generation is optional; use the first prompt when the hidden request fails.
  }
  return fallbackTitle(conversation.messages.find((message) => message.role === "user")?.content ?? "New chat");
}

function fallbackTitle(content: string): string {
  const title = content.trim().replace(/\s+/g, " ").slice(0, 80);
  return title || "New chat";
}

function splitThinking(content: string): { content: string; reasoning?: string } {
  const match = content.match(/^\s*<think>([\s\S]*?)<\/think>\s*/i);
  return match ? { content: content.slice(match[0].length), reasoning: match[1].trim() } : { content };
}

function sortSummaries(summaries: ChatConversationSummary[]): ChatConversationSummary[] {
  return summaries.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt));
}

function chatId(): string {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function messageId(): string {
  return chatId();
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
