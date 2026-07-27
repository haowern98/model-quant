export function ChatEditor() {
  return (
    <section className="chat-editor" aria-label="New chat">
      <div className="chat-editor-messages" />
      <form className="chat-composer" onSubmit={(event) => event.preventDefault()}>
        <textarea aria-label="Message" placeholder="Send a message to the model..." />
        <div className="chat-composer-actions">
          <button type="button" className="chat-composer-action" disabled aria-label="Attach files">
            <span className="codicon codicon-add" aria-hidden="true" />
          </button>
          <button type="submit" className="chat-composer-send" disabled aria-label="Send message">
            <span className="codicon codicon-arrow-up" aria-hidden="true" />
          </button>
        </div>
      </form>
    </section>
  );
}
