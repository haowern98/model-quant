import { useEffect, useRef, useState, type KeyboardEvent as ReactKeyboardEvent, type PointerEvent as ReactPointerEvent } from "react";
import { type ChatMessageData, ChatMessage } from "./ChatMessage";
import { cancelChatGeneration, type ChatTracePayload } from "../../lib/tauri-bridge";
import { ChatTracePanel } from "./ChatTracePanel";

const TRACE_PANEL_MIN_WIDTH = 520;
const CHAT_MAIN_MIN_WIDTH = 320;

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
  onCloseTrace: () => void;
  onTraceTokenChange: (index: number) => void;
  onSend: (content: string) => void;
}

export function ChatEditor({ messages, draft, modelReady, sending, disabled, error, traceArmed, tracePanelOpen, traceMessageId, tracePayload, traceTokenIndex, traceLoading, traceError, onDraftChange, onTraceArmedChange, onOpenTrace, onCloseTrace, onTraceTokenChange, onSend }: ChatEditorProps) {
  const editorRef = useRef<HTMLElement>(null);
  const [tracePanelWidth, setTracePanelWidth] = useState(TRACE_PANEL_MIN_WIDTH);
  const [tracePanelFullscreen, setTracePanelFullscreen] = useState(false);

  useEffect(() => {
    if (tracePanelOpen) return;
    setTracePanelWidth(TRACE_PANEL_MIN_WIDTH);
    setTracePanelFullscreen(false);
  }, [tracePanelOpen]);

  const tracePanelMaxSplitWidth = () => {
    const editorWidth = editorRef.current?.getBoundingClientRect().width ?? 1280;
    return Math.max(TRACE_PANEL_MIN_WIDTH, Math.floor(editorWidth - CHAT_MAIN_MIN_WIDTH));
  };

  const closeTracePanel = () => {
    setTracePanelFullscreen(false);
    setTracePanelWidth(TRACE_PANEL_MIN_WIDTH);
    onCloseTrace();
  };

  const setTracePanelSplitWidth = (requestedWidth: number) => {
    if (requestedWidth < TRACE_PANEL_MIN_WIDTH) {
      closeTracePanel();
      return;
    }
    if (requestedWidth >= tracePanelMaxSplitWidth()) {
      setTracePanelFullscreen(true);
      return;
    }
    setTracePanelFullscreen(false);
    setTracePanelWidth(requestedWidth);
  };

  const startTraceResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = tracePanelFullscreen
      ? (editorRef.current?.getBoundingClientRect().width ?? tracePanelMaxSplitWidth())
      : tracePanelWidth;
    let closed = false;
    document.body.classList.add("resizing-trace");

    const handleMove = (moveEvent: PointerEvent) => {
      if (closed) return;
      const requestedWidth = startWidth + startX - moveEvent.clientX;
      if (requestedWidth < TRACE_PANEL_MIN_WIDTH) closed = true;
      setTracePanelSplitWidth(requestedWidth);
    };
    const stopResize = () => {
      document.body.classList.remove("resizing-trace");
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", stopResize);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", stopResize);
  };

  const handleTraceResizeKey = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    event.preventDefault();
    if (tracePanelFullscreen && event.key === "ArrowRight") {
      setTracePanelFullscreen(false);
      setTracePanelWidth(Math.max(TRACE_PANEL_MIN_WIDTH, tracePanelMaxSplitWidth() - 10));
      return;
    }
    const currentWidth = tracePanelFullscreen
      ? (editorRef.current?.getBoundingClientRect().width ?? tracePanelMaxSplitWidth())
      : tracePanelWidth;
    setTracePanelSplitWidth(currentWidth + (event.key === "ArrowLeft" ? 10 : -10));
  };

  return (
    <section
      ref={editorRef}
      className={`chat-editor${tracePanelOpen ? " chat-editor-trace-open" : ""}${tracePanelFullscreen ? " chat-editor-trace-fullscreen" : ""}`}
      aria-label="New chat"
      style={tracePanelOpen && !tracePanelFullscreen ? { gridTemplateColumns: `minmax(0, 1fr) ${tracePanelWidth}px` } : undefined}
    >
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
      {tracePanelOpen ? (
        <>
          <div
            className="resize-handle chat-trace-resizer"
            role="separator"
            aria-label="Resize Logit Lens"
            aria-orientation="vertical"
            aria-valuemin={TRACE_PANEL_MIN_WIDTH}
            aria-valuemax={tracePanelMaxSplitWidth()}
            aria-valuenow={Math.round(tracePanelFullscreen ? (editorRef.current?.getBoundingClientRect().width ?? tracePanelMaxSplitWidth()) : tracePanelWidth)}
            tabIndex={0}
            style={tracePanelFullscreen ? undefined : { right: tracePanelWidth }}
            onPointerDown={startTraceResize}
            onKeyDown={handleTraceResizeKey}
          />
          <ChatTracePanel trace={tracePayload} selectedTokenIndex={traceTokenIndex} onSelectTokenIndex={onTraceTokenChange} loading={traceLoading} error={traceError} />
        </>
      ) : null}
    </section>
  );
}
