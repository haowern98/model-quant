import { type ChatMessageData, ChatMessage } from "./ChatMessage";
import { cancelChatGeneration } from "../../lib/tauri-bridge";

interface ChatEditorProps {
  messages: ChatMessageData[];
  draft: string;
  modelReady: boolean;
  sending: boolean;
  disabled: boolean;
  error: string | null;
  onDraftChange: (draft: string) => void;
  onSend: (content: string) => void;
}

export function ChatEditor({ messages, draft, modelReady, sending, disabled, error, onDraftChange, onSend }: ChatEditorProps) {
  return (
    <section className="chat-editor" aria-label="New chat">
      <div className="chat-editor-messages">
        {messages.map((message) => <ChatMessage key={message.id} message={message} />)}
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
          <button type="button" className="chat-composer-action" disabled aria-label="Attach files">
            <span className="codicon codicon-add" aria-hidden="true" />
          </button>
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
    </section>
  );
}
