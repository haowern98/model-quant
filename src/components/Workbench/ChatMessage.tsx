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
};

export function ChatMessage({ message }: { message: ChatMessageData }) {
  return (
    <article className={`chat-message chat-message-${message.role}`}>
      <div className="chat-message-content">{message.content}</div>
      {message.reasoning ? (
        <details className="chat-message-reasoning">
          <summary>Thinking</summary>
          <div>{message.reasoning}</div>
        </details>
      ) : null}
      {message.role === "assistant" && message.model ? (
        <div className="chat-message-metadata">
          <span>{message.model}</span>
          {message.tokensPerSecond !== undefined ? <span>{message.tokensPerSecond.toFixed(2)} tok/sec</span> : null}
          {message.promptTokens !== undefined ? <span>{message.promptTokens.toLocaleString()} prompt tokens</span> : null}
          {message.durationSeconds !== undefined ? <span>{message.durationSeconds.toFixed(2)} s</span> : null}
          {message.finishReason ? <span>Stop reason: {message.finishReason}</span> : null}
        </div>
      ) : null}
    </article>
  );
}
