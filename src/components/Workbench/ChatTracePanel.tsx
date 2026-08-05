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
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const selectedToken = trace?.tokens[selectedTokenIndex] ?? null;
  const layers = selectedToken?.layers ?? [];
  const finalCandidates = layers.at(-1)?.candidates ?? [];
  const columns = finalCandidates.slice(0, candidateCount);
  const selectedLayerData = layers.find((layer) => layer.layer === selectedLayer) ?? layers.at(-1) ?? null;
  const selectedCandidate = selectedLayerData?.candidates.find((candidate) => candidate.tokenId === selectedCandidateId) ?? null;
  const selectedRank = selectedCandidate && selectedLayerData
    ? selectedLayerData.candidates.findIndex((candidate) => candidate.tokenId === selectedCandidate.tokenId) + 1
    : null;

  useEffect(() => {
    setSelectedLayer(layers.at(-1)?.layer ?? null);
    setSelectedCandidateId(finalCandidates[0]?.tokenId ?? null);
  }, [selectedTokenIndex, trace]);

  if (loading) {
    return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">Loading trace…</aside>;
  }
  if (error) {
    return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">{error}</aside>;
  }
  if (!trace || !trace.supported || !selectedToken || layers.length === 0) {
    return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">This message has no supported trace.</aside>;
  }

  return (
    <aside className="chat-trace-panel" aria-label="Logit Lens">
      <div className="chat-trace-tabs" aria-label="Trace views">
        <span className="chat-trace-tab-active">Logit Lens</span>
      </div>
      <div className="chat-trace-controls">
        <label className="chat-trace-control">
          <span>Inspecting position</span>
          <select value={selectedTokenIndex} onChange={(event) => onSelectTokenIndex(Number(event.currentTarget.value))}>
            {trace.tokens.map((token, index) => <option key={`${token.index}-${token.tokenId}`} value={index}>{`After token #${token.index}: ${displayToken(token.tokenText)}`}</option>)}
          </select>
        </label>
        <label className="chat-trace-control">
          <span>View</span>
          <select value="logit" disabled><option value="logit">Logit Lens (Logit)</option></select>
        </label>
        <label className="chat-trace-control">
          <span>Candidate tokens</span>
          <select value={candidateCount} onChange={(event) => setCandidateCount(Number(event.currentTarget.value) as typeof candidateCount)}>
            {candidateCounts.map((count) => <option key={count} value={count}>{`Top ${count} (final layer)`}</option>)}
          </select>
        </label>
      </div>
      <p className="chat-trace-caption">Stored top-64 logits. A dash means that token was outside that layer’s captured top 64.</p>
      <div className="chat-trace-table-wrap">
        <table className="chat-trace-table">
          <thead><tr><th>Layer</th>{columns.map((candidate) => <th key={candidate.tokenId}>{displayToken(candidate.tokenText)}</th>)}</tr></thead>
          <tbody>
            {layers.map((layer) => (
              <tr key={layer.layer} className={layer.layer === selectedLayer ? "chat-trace-selected-row" : ""}>
                <th><button type="button" onClick={() => setSelectedLayer(layer.layer)}>{layer.layer}</button></th>
                {columns.map((column) => {
                  const candidate = layer.candidates.find((item) => item.tokenId === column.tokenId);
                  return <td key={`${layer.layer}-${column.tokenId}`}><button type="button" disabled={!candidate} onClick={() => { setSelectedLayer(layer.layer); setSelectedCandidateId(column.tokenId); }}>{candidate ? candidate.logit.toFixed(3) : "—"}</button></td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="chat-trace-selection">Selected: layer {selectedLayer ?? "—"} × {selectedCandidate ? `‘${displayToken(selectedCandidate.tokenText)}’` : "no captured candidate"}</div>
      <div className="chat-trace-details">
        <div><span>Generated token</span><strong>{displayToken(selectedToken.tokenText)}</strong><span>Token ID</span><strong>{selectedToken.tokenId}</strong><span>Position</span><strong>{`After token #${selectedToken.index}`}</strong></div>
        <div><span>Logit Lens</span><strong>{selectedCandidate ? selectedCandidate.logit.toFixed(3) : "—"}</strong><span>Top-64 rank</span><strong>{selectedRank ?? "—"}</strong><span>Captured candidates</span><strong>{selectedLayerData?.candidates.length ?? 0}</strong></div>
        <div className="chat-trace-predictions"><span>Top predictions at this layer</span>{selectedLayerData?.candidates.slice(0, 5).map((candidate, index) => <strong key={candidate.tokenId}>{`${index + 1}. ${displayToken(candidate.tokenText)}`} <i style={{ width: barWidth(candidate, selectedLayerData.candidates) }} /> {candidate.logit.toFixed(3)}</strong>)}</div>
      </div>
      <div className="chat-trace-footer">Trace buffer: {trace.tokens.length} generated tokens · {layers.length} layers · 64 candidates/layer</div>
    </aside>
  );
}

function displayToken(token: string): string {
  return token.replace(/\n/g, "↵").replace(/ /g, "·") || "∅";
}

function barWidth(candidate: ChatTraceCandidate, candidates: ChatTraceCandidate[]): string {
  const max = candidates[0]?.logit ?? candidate.logit;
  const min = candidates.at(-1)?.logit ?? candidate.logit;
  const ratio = max === min ? 1 : (candidate.logit - min) / (max - min);
  return `${Math.round(8 + ratio * 92)}%`;
}
