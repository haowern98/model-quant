import { useEffect, useState } from "react";
import type { ChatTraceCandidate, ChatTracePayload } from "../../lib/tauri-bridge";

type ChatTracePanelProps = {
  trace: ChatTracePayload | null;
  selectedTokenIndex: number;
  onSelectTokenIndex: (index: number) => void;
  loading: boolean;
  error: string | null;
};

const candidateCounts = [12, 24, 64] as const;

export function ChatTracePanel({ trace, selectedTokenIndex, onSelectTokenIndex, loading, error }: ChatTracePanelProps) {
  const [candidateCount, setCandidateCount] = useState<(typeof candidateCounts)[number]>(12);
  const [view, setView] = useState<"logit" | "probability">("logit");
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const selectedToken = trace?.tokens[selectedTokenIndex] ?? null;
  const layers = selectedToken?.layers ?? [];
  const selectedLayerData = layers.find((layer) => layer.layer === selectedLayer) ?? layers.at(-1) ?? null;
  const selectedCandidate = selectedLayerData?.candidates.find((candidate) => candidate.tokenId === selectedCandidateId) ?? null;
  const selectedRank = selectedCandidate && selectedLayerData
    ? selectedLayerData.candidates.findIndex((candidate) => candidate.tokenId === selectedCandidate.tokenId) + 1
    : null;
  const hasProbabilities = layers.every((layer) => layer.candidates.every((candidate) => typeof candidate.probability === "number"));

  useEffect(() => {
    const finalLayer = layers.at(-1);
    setSelectedLayer(finalLayer?.layer ?? null);
    setSelectedCandidateId(finalLayer?.candidates[0]?.tokenId ?? null);
    if (!hasProbabilities) setView("logit");
  }, [selectedTokenIndex, trace]);

  if (loading) return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">Loading trace…</aside>;
  if (error) return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">{error}</aside>;
  if (!trace || !trace.supported || !selectedToken || layers.length === 0) {
    return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">This message has no supported trace.</aside>;
  }

  return (
    <aside className="chat-trace-panel" aria-label="Logit Lens">
      <div className="chat-trace-tabs" aria-label="Trace views"><span className="chat-trace-tab-active">Logit Lens</span></div>
      <div className="chat-trace-controls">
        <label className="chat-trace-control">
          <span>Inspecting position</span>
          <select value={selectedTokenIndex} onChange={(event) => onSelectTokenIndex(Number(event.currentTarget.value))}>
            {trace.tokens.map((token, index) => <option key={`${token.index}-${token.tokenId}`} value={index}>{`Token #${token.index}: ${displayToken(token.tokenText)}`}</option>)}
          </select>
        </label>
        <label className="chat-trace-control">
          <span>View</span>
          <select value={view} onChange={(event) => setView(event.currentTarget.value as typeof view)}>
            <option value="logit">Logit Lens (Logit)</option>
            <option value="probability" disabled={!hasProbabilities}>Logit Lens (Probability)</option>
          </select>
        </label>
        <label className="chat-trace-control">
          <span>Candidate tokens</span>
          <select value={candidateCount} onChange={(event) => setCandidateCount(Number(event.currentTarget.value) as typeof candidateCount)}>
            {candidateCounts.map((count) => <option key={count} value={count}>{`Top ${count}`}</option>)}
          </select>
        </label>
      </div>
      <p className="chat-trace-title">Predictions for token #{selectedToken.index}: “{displayToken(selectedToken.tokenText)}”</p>
      <p className="chat-trace-caption">
        {hasProbabilities
          ? "The outlined token is the actual generated token when it appears in this layer’s displayed top candidates."
          : "This older trace has logits only. Regenerate this reply with Trace On to view exact probabilities."}
      </p>
      <div className="chat-trace-table-wrap">
        <table className="chat-trace-table">
          <thead><tr><th>Layer</th>{Array.from({ length: candidateCount }, (_, index) => <th key={index}>Rank {index + 1}</th>)}</tr></thead>
          <tbody>
            {layers.map((layer) => (
              <tr key={layer.layer} className={layer.layer === selectedLayer ? "chat-trace-selected-row" : ""}>
                <th><button type="button" onClick={() => setSelectedLayer(layer.layer)}>L{layer.layer}</button></th>
                {layer.candidates.slice(0, candidateCount).map((candidate) => (
                  <td key={`${layer.layer}-${candidate.tokenId}`}>
                    <button
                      type="button"
                      className={candidate.tokenId === selectedToken.tokenId ? "chat-trace-generated-token" : ""}
                      onClick={() => { setSelectedLayer(layer.layer); setSelectedCandidateId(candidate.tokenId); }}
                    >
                      <span className="chat-trace-cell-token">{displayToken(candidate.tokenText)}</span>
                      <span className="chat-trace-cell-value">{metricValue(candidate, view)}</span>
                    </button>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="chat-trace-selection">Selected: layer {selectedLayer ?? "—"} × {selectedCandidate ? `‘${displayToken(selectedCandidate.tokenText)}’` : "no candidate"}</div>
      <div className="chat-trace-details">
        <div><span>Generated token</span><strong>{displayToken(selectedToken.tokenText)}</strong><span>Token ID</span><strong>{selectedToken.tokenId}</strong><span>Position</span><strong>{`After token #${selectedToken.index}`}</strong></div>
        <div><span>{view === "logit" ? "Logit Lens" : "Probability"}</span><strong>{selectedCandidate ? metricValue(selectedCandidate, view) : "—"}</strong><span>Top-64 rank</span><strong>{selectedRank ?? "—"}</strong><span>Captured candidates</span><strong>{selectedLayerData?.candidates.length ?? 0}</strong></div>
        <div className="chat-trace-predictions"><span>Top predictions at this layer</span>{selectedLayerData?.candidates.slice(0, 5).map((candidate, index) => <strong key={candidate.tokenId}>{`${index + 1}. ${displayToken(candidate.tokenText)}  ${metricValue(candidate, view)}`}</strong>)}</div>
      </div>
      <div className="chat-trace-footer">Trace buffer: {trace.tokens.length} generated tokens · {layers.length} layers · 64 candidates/layer</div>
    </aside>
  );
}

function displayToken(token: string): string {
  return token.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
}

function metricValue(candidate: ChatTraceCandidate, view: "logit" | "probability"): string {
  return view === "probability" && typeof candidate.probability === "number"
    ? candidate.probability.toFixed(4)
    : candidate.logit.toFixed(3);
}
