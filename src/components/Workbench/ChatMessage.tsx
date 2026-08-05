import type { ChatMessageData } from "./chat/chatTypes";

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
  const tracedContent = traceTokenTexts?.join("") === message.content;

  return (
    <article className={`chat-message chat-message-${message.role}`}>
      {message.reasoning ? (
        <details className="chat-message-reasoning">
          <summary>Thinking</summary>
          <div>{message.reasoning}</div>
        </details>
      ) : null}
      <div className="chat-message-content">
        {tracedContent && traceTokenTexts && onSelectTraceToken
          ? traceTokenTexts.map((token, index) => (
            <button
              type="button"
              key={`${index}-${token}`}
              className={`chat-trace-token${index === selectedTraceTokenIndex ? " chat-trace-token-selected" : ""}`}
              onClick={() => onSelectTraceToken(index)}
            >
              {token}
            </button>
          ))
          : message.content}
      </div>
      {message.role === "assistant" && message.model ? (
        <div className="chat-message-metadata">
          <span>{message.model}</span>
          {message.tokensPerSecond !== undefined ? <span>{message.tokensPerSecond.toFixed(2)} tok/sec</span> : null}
          {message.promptTokens !== undefined ? <span>{message.promptTokens.toLocaleString()} prompt tokens</span> : null}
          {message.durationSeconds !== undefined ? <span>{message.durationSeconds.toFixed(2)} s</span> : null}
          {message.finishReason ? <span>Stop reason: {message.finishReason}</span> : null}
          {message.trace ? <button type="button" className="chat-message-trace-button" onClick={onOpenTrace}>Trace saved</button> : null}
        </div>
      ) : null}
    </article>
  );
}
