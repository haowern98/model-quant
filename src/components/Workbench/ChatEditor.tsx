import { type ChatMessageData, ChatMessage } from "./ChatMessage";
import { cancelChatGeneration } from "../../lib/tauri-bridge";

const TRACE_COLUMNS = ["orbit", "alignment", "inclination", "rotation", "tilt", "angle"];
const TRACE_ROWS: Array<[string, string[]]> = [
  ["4", ["0.02", "0.01", "0.03", "0.01", "0.02", "0.01"]],
  ["8", ["0.01", "0.02", "0.02", "0.03", "0.01", "0.01"]],
  ["12", ["0.02", "0.04", "0.01", "0.06", "0.02", "0.01"]],
  ["16", ["0.03", "0.06", "0.10", "0.18", "0.01", "0.01"]],
  ["20", ["0.04", "0.08", "0.10", "0.18", "0.01", "0.01"]],
  ["24", ["0.04", "0.07", "0.17", "0.19", "0.00", "0.02"]],
  ["28", ["0.05", "0.05", "0.54", "0.18", "0.01", "0.03"]],
  ["32", ["0.06", "0.02", "0.31", "0.13", "0.01", "0.01"]],
  ["36", ["0.06", "0.01", "0.39", "0.09", "0.02", "0.03"]],
];

interface ChatEditorProps {
  messages: ChatMessageData[];
  draft: string;
  modelReady: boolean;
  sending: boolean;
  disabled: boolean;
  error: string | null;
  tracePanelOpen: boolean;
  onDraftChange: (draft: string) => void;
  onTracePanelOpenChange: (open: boolean) => void;
  onSend: (content: string) => void;
}

export function ChatEditor({ messages, draft, modelReady, sending, disabled, error, tracePanelOpen, onDraftChange, onTracePanelOpenChange, onSend }: ChatEditorProps) {
  return (
    <section className={`chat-editor${tracePanelOpen ? " chat-editor-trace-open" : ""}`} aria-label="New chat">
      <div className="chat-editor-main">
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
            <div className="chat-composer-leading-actions">
              <button type="button" className="chat-composer-action" disabled aria-label="Attach files">
                <span className="codicon codicon-add" aria-hidden="true" />
              </button>
              <button
                type="button"
                className="chat-composer-trace-toggle"
                aria-label={`Trace: ${tracePanelOpen ? "On" : "Off"}`}
                aria-pressed={tracePanelOpen}
                onClick={() => onTracePanelOpenChange(!tracePanelOpen)}
              >
                <span>{`Trace: ${tracePanelOpen ? "On" : "Off"}`}</span>
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
      {tracePanelOpen ? <ChatTracePanel /> : null}
    </section>
  );
}

function ChatTracePanel() {
  return (
    <aside className="chat-trace-panel" aria-label="Trace & Mechanistic Analysis">
      <h2>TRACE &amp; MECHANISTIC ANALYSIS</h2>
      <div className="chat-trace-controls">
        <TraceControl label="Inspecting position" value="After token #9 (every)" />
        <TraceControl label="View" value="Logit Lens (Probability)" />
        <TraceControl label="Candidate tokens" value="Auto (Top 12)" />
      </div>
      <div className="chat-trace-tabs" aria-label="Trace views">
        <span className="chat-trace-tab-active">Evolution</span>
        <span>Top Predictions</span>
        <span>Residual Stream</span>
        <span>Raw</span>
      </div>
      <div className="chat-trace-metrics">
        <span>Metric:</span>
        <span className="chat-trace-metric-active">Probability</span>
        <span>Logit</span>
        <span>Rank</span>
      </div>
      <div className="chat-trace-scale" aria-hidden="true">
        <span>0</span><span>0.25</span><span>0.50</span><span>0.75</span><span>1.00</span>
      </div>
      <div className="chat-trace-table-wrap">
        <table className="chat-trace-table">
          <thead><tr><th>Layer</th>{TRACE_COLUMNS.map((column) => <th key={column}>{column}</th>)}</tr></thead>
          <tbody>
            {TRACE_ROWS.map(([layer, values]) => (
              <tr key={layer} className={layer === "28" ? "chat-trace-selected-row" : ""}>
                <th>{layer}</th>
                {values.map((value, index) => <td key={`${layer}-${TRACE_COLUMNS[index]}`} className={index === 2 ? "chat-trace-heat" : ""}>{value}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="chat-trace-selection">Selected: Layer 28 × ‘inclination’</div>
      <div className="chat-trace-details">
        <div><span>Token</span><strong>inclination</strong><span>Token ID</span><strong>12345</strong><span>Position</span><strong>After token #9</strong></div>
        <div><span>Logit Lens (Probability)</span><strong>54.3%</strong><span>Rank</span><strong>1</strong><span>Logit</span><strong>8.42</strong></div>
        <div className="chat-trace-predictions"><span>Top Predictions at this Layer</span><strong>1. inclination <i style={{ width: "54%" }} /> 54.3%</strong><strong>2. alignment <i style={{ width: "18%" }} /> 17.8%</strong><strong>3. tilt <i style={{ width: "8%" }} /> 8.2%</strong></div>
      </div>
      <div className="chat-trace-actions"><button type="button">Run Trace</button><button type="button">Refresh</button><button type="button">Download</button><button type="button">Clear</button><span>Trace buffer: no trace</span></div>
    </aside>
  );
}

function TraceControl({ label, value }: { label: string; value: string }) {
  return <div className="chat-trace-control"><span>{label}</span><strong>{value}<i className="codicon codicon-chevron-down" aria-hidden="true" /></strong></div>;
}
