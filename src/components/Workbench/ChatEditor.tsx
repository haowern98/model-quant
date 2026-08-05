import { type ChatMessageData, ChatMessage } from "./ChatMessage";
import { cancelChatGeneration, type ChatTracePayload } from "../../lib/tauri-bridge";
import { ChatTracePanel } from "./ChatTracePanel";

interface ChatEditorProps {
  messages: ChatMessageData[];
  draft: string;
  modelReady: boolean;
  sending: boolean;
  disabled: boolean;
  error: string | null;
  traceArmed: boolean;
  tracePanelOpen: boolean;
  traceMessageId: string | null;
  tracePayload: ChatTracePayload | null;
  traceTokenIndex: number;
  traceLoading: boolean;
  traceError: string | null;
  onDraftChange: (draft: string) => void;
  onTraceArmedChange: (armed: boolean) => void;
  onOpenTrace: (messageId: string) => void;
  onTraceTokenChange: (index: number) => void;
  onSend: (content: string) => void;
}

export function ChatEditor({ messages, draft, modelReady, sending, disabled, error, traceArmed, tracePanelOpen, traceMessageId, tracePayload, traceTokenIndex, traceLoading, traceError, onDraftChange, onTraceArmedChange, onOpenTrace, onTraceTokenChange, onSend }: ChatEditorProps) {
  return (
    <section className={`chat-editor${tracePanelOpen ? " chat-editor-trace-open" : ""}`} aria-label="New chat">
      <div className="chat-editor-main">
        <div className="chat-editor-messages">
          {messages.map((message) => <ChatMessage
            key={message.id}
            message={message}
            traceTokenTexts={message.id === traceMessageId ? tracePayload?.tokens.map((token) => token.tokenText) : undefined}
            selectedTraceTokenIndex={message.id === traceMessageId ? traceTokenIndex : undefined}
            onOpenTrace={message.trace ? () => onOpenTrace(message.id) : undefined}
            onSelectTraceToken={message.id === traceMessageId && tracePayload?.supported ? onTraceTokenChange : undefined}
          />)}
          {error ? <p className="chat-editor-error" role="alert">{error}</p> : null}
        </div>
        <form
          className="chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
            if (!draft.trim() || sending || disabled) return;
            onSend(draft);
            onDraftChange("");
          }}
        >
          <textarea
            aria-label="Message"
            placeholder="Send a message to the model..."
            value={draft}
            disabled={disabled || sending}
            onChange={(event) => onDraftChange(event.currentTarget.value)}
          />
          <div className="chat-composer-actions">
            <div className="chat-composer-leading-actions">
              <button type="button" className="chat-composer-action" disabled aria-label="Attach files">
                <span className="codicon codicon-add" aria-hidden="true" />
              </button>
              <button
                type="button"
                className="chat-composer-trace-toggle"
                aria-label={`Trace: ${traceArmed ? "On" : "Off"}`}
                aria-pressed={traceArmed}
                onClick={() => onTraceArmedChange(!traceArmed)}
              >
                <span>{`Trace: ${traceArmed ? "On" : "Off"}`}</span>
                <span className="codicon codicon-settings-gear" aria-hidden="true" />
              </button>
            </div>
            <button
              type={sending ? "button" : "submit"}
              className={`chat-composer-send${modelReady ? " chat-composer-send-ready" : ""}`}
              disabled={sending ? false : disabled || !draft.trim()}
              aria-label={sending ? "Cancel generation" : "Send message"}
              onClick={sending ? () => { void cancelChatGeneration(); } : undefined}
            >
              <span
                className={sending ? "chat-composer-stop-icon" : "codicon codicon-arrow-up"}
                aria-hidden="true"
              />
            </button>
          </div>
        </form>
      </div>
      {tracePanelOpen ? <ChatTracePanel trace={tracePayload} selectedTokenIndex={traceTokenIndex} onSelectTokenIndex={onTraceTokenChange} loading={traceLoading} error={traceError} /> : null}
    </section>
  );
}
