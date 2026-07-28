import { type ChatMessageData, ChatMessage } from "./ChatMessage";

interface ChatEditorProps {
  messages: ChatMessageData[];
  draft: string;
  sending: boolean;
  disabled: boolean;
  error: string | null;
  onDraftChange: (draft: string) => void;
  onSend: (content: string) => void;
}

export function ChatEditor({ messages, draft, sending, disabled, error, onDraftChange, onSend }: ChatEditorProps) {
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
          <button type="submit" className="chat-composer-send" disabled={disabled || sending || !draft.trim()} aria-label="Send message">
            <span className="codicon codicon-arrow-up" aria-hidden="true" />
          </button>
        </div>
      </form>
    </section>
  );
}
