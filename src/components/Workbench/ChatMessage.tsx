import { Fragment } from "react";
import type { ChatMessageData } from "./chat/chatTypes";
import { mapTraceText, type TraceTextPiece } from "./chat/traceText";

export type { ChatMessageData } from "./chat/chatTypes";

type ChatMessageProps = {
  message: ChatMessageData;
  traceTokenTexts?: string[];
  selectedTraceTokenIndex?: number;
  onOpenTrace?: () => void;
  onSelectTraceToken?: (index: number) => void;
};

export function ChatMessage({
  message,
  traceTokenTexts,
  selectedTraceTokenIndex,
  onOpenTrace,
  onSelectTraceToken,
}: ChatMessageProps) {
  const rawTraceText = traceTokenTexts?.join("");
  const reasoningStart = message.reasoning && rawTraceText
    ? rawTraceText.indexOf(message.reasoning)
    : -1;
  const reasoningTrace = traceTokenTexts && message.reasoning
    ? mapTraceText(traceTokenTexts, message.reasoning)
    : undefined;
  const contentTrace = traceTokenTexts
    ? mapTraceText(
      traceTokenTexts,
      message.content,
      reasoningStart >= 0 ? reasoningStart + message.reasoning!.length : 0,
    )
    : undefined;

  const traceText = (pieces: TraceTextPiece[] | undefined, fallback: string) => {
    if (!pieces || !onSelectTraceToken) return fallback;
    return pieces.flatMap((piece) => piece.text
      .split(/(\s+)/)
      .filter((part) => part.length > 0)
      .map((part, partIndex) => {
        const tokenIndex = piece.tokenIndex;
        if (/^\s+$/.test(part) || tokenIndex === null) {
          return <Fragment key={`${tokenIndex}-${partIndex}`}>{part}</Fragment>;
        }
        return (
          <button
            type="button"
            key={`${tokenIndex}-${partIndex}`}
            className={`chat-trace-token${tokenIndex === selectedTraceTokenIndex ? " chat-trace-token-selected" : ""}`}
            onClick={() => onSelectTraceToken(tokenIndex)}
          >
            {part}
          </button>
        );
      }));
  };

  return (
    <article className={`chat-message chat-message-${message.role}`}>
      {message.reasoning ? (
        <details className="chat-message-reasoning">
          <summary>Thinking</summary>
          <div>{traceText(reasoningTrace, message.reasoning)}</div>
        </details>
      ) : null}
      <div className="chat-message-content">
        {traceText(contentTrace, message.content)}
      </div>
      {message.role === "assistant" && message.model ? (
        <div className="chat-message-metadata">
          <span>{message.model}</span>
          {message.tokensPerSecond !== undefined ? <span>{message.tokensPerSecond.toFixed(2)} tok/sec</span> : null}
          {message.promptTokens !== undefined ? <span>{message.promptTokens.toLocaleString()} prompt tokens</span> : null}
          {message.durationSeconds !== undefined ? <span>{message.durationSeconds.toFixed(2)} s</span> : null}
          {message.finishReason ? <span>Stop reason: {message.finishReason}</span> : null}
          {message.trace?.status === "saving" ? <span className="chat-message-trace-button">Saving trace…</span> : null}
          {message.trace?.status === "failed" ? <span className="chat-message-trace-button">Trace unavailable</span> : null}
          {message.trace && message.trace.status !== "saving" && message.trace.status !== "failed" ? (
            <button type="button" className="chat-message-trace-button" onClick={onOpenTrace}>Trace saved</button>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}
