import { useEffect, useId, useLayoutEffect, useRef, useState, type CSSProperties } from "react";
import type { ChatTraceCandidate, ChatTracePayload } from "../../lib/tauri-bridge";

type ChatTracePanelProps = {
  trace: ChatTracePayload | null;
  selectedTokenIndex: number;
  onSelectTokenIndex: (index: number) => void;
  loading: boolean;
  error: string | null;
  maximized?: boolean;
  onToggleMaximized?: () => void;
  onClose?: () => void;
};

const candidateCounts = [12, 24, 64] as const;

type TraceDropdownOption<T extends string | number> = {
  value: T;
  label: string;
  disabled?: boolean;
};

function TraceDropdown<T extends string | number>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: TraceDropdownOption<T>[];
  onChange: (value: T) => void;
}) {
  const [open, setOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const listboxId = useId();
  const selected = options.find((option) => option.value === value);

  useEffect(() => {
    if (!open) return undefined;
    const closeOnOutsidePress = (event: PointerEvent) => {
      if (!containerRef.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("pointerdown", closeOnOutsidePress);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsidePress);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  return (
    <div className="chat-trace-control" ref={containerRef}>
      <span>{label}</span>
      <button
        type="button"
        className="chat-trace-select-trigger"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={listboxId}
        onClick={() => setOpen((current) => !current)}
      >
        <span>{selected?.label}</span>
        <span className="codicon codicon-chevron-down" aria-hidden="true" />
      </button>
      {open ? (
        <div id={listboxId} className="chat-trace-select-menu" role="listbox" aria-label={label}>
          {options.map((option) => (
            <button
              type="button"
              role="option"
              key={option.value}
              aria-selected={option.value === value}
              disabled={option.disabled}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
            >
              {option.label}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function ChatTracePanel({ trace, selectedTokenIndex, onSelectTokenIndex, loading, error, maximized = false, onToggleMaximized, onClose }: ChatTracePanelProps) {
  const [candidateCount, setCandidateCount] = useState<(typeof candidateCounts)[number]>(12);
  const [view, setView] = useState<"logit" | "probability">("logit");
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null);
  const [selectedCandidateId, setSelectedCandidateId] = useState<number | null>(null);
  const [gridScrollWidth, setGridScrollWidth] = useState(0);
  const gridWrapRef = useRef<HTMLDivElement>(null);
  const gridScrollbarRef = useRef<HTMLDivElement>(null);
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

  useLayoutEffect(() => {
    const gridWrap = gridWrapRef.current;
    if (!gridWrap) return undefined;

    const updateScrollWidth = () => setGridScrollWidth(gridWrap.scrollWidth);
    updateScrollWidth();
    const observer = typeof ResizeObserver === "undefined" ? null : new ResizeObserver(updateScrollWidth);
    observer?.observe(gridWrap);
    window.addEventListener("resize", updateScrollWidth);
    return () => {
      observer?.disconnect();
      window.removeEventListener("resize", updateScrollWidth);
    };
  }, [candidateCount, selectedTokenIndex, trace]);

  const syncGridScroll = (source: "grid" | "scrollbar") => {
    const gridWrap = gridWrapRef.current;
    const scrollbar = gridScrollbarRef.current;
    if (!gridWrap || !scrollbar) return;
    if (source === "grid") scrollbar.scrollLeft = gridWrap.scrollLeft;
    else gridWrap.scrollLeft = scrollbar.scrollLeft;
  };

  if (loading) return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">Loading trace…</aside>;
  if (error) return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">{error}</aside>;
  if (!trace || !trace.supported || !selectedToken || layers.length === 0) {
    return <aside className="chat-trace-panel chat-trace-empty" aria-label="Logit Lens">This message has no supported trace.</aside>;
  }

  const traceGridStyle: CSSProperties = {
    gridTemplateColumns: `58px repeat(${candidateCount}, minmax(80px, 1fr))`,
    gridTemplateRows: `28px repeat(${layers.length}, 44px)`,
    minWidth: `${58 + candidateCount * 80}px`,
  };

  return (
    <aside className="chat-trace-panel" aria-label="Logit Lens">
      <div className="chat-trace-tabs" aria-label="Trace views">
        <span className="chat-trace-tab-active">LOGIT LENS</span>
        {onToggleMaximized && onClose ? (
          <div className="chat-trace-tab-actions">
            <button
              type="button"
              className="chat-trace-panel-action"
              aria-label={maximized ? "Restore trace panel" : "Maximize trace panel"}
              title={maximized ? "Restore trace panel" : "Maximize trace panel"}
              onClick={onToggleMaximized}
            >
              <span className={`codicon codicon-${maximized ? "screen-normal" : "screen-full"}`} aria-hidden="true" />
            </button>
            <button
              type="button"
              className="chat-trace-panel-action"
              aria-label="Close trace panel"
              title="Close trace panel"
              onClick={onClose}
            >
              <span className="codicon codicon-close" aria-hidden="true" />
            </button>
          </div>
        ) : null}
      </div>
      <div className="chat-trace-controls">
        <TraceDropdown
          label="Inspecting position"
          value={selectedTokenIndex}
          options={trace.tokens.map((token, index) => ({
            value: index,
            label: `Token #${token.index}: ${displayToken(token.tokenText)}`,
          }))}
          onChange={onSelectTokenIndex}
        />
        <TraceDropdown
          label="View"
          value={view}
          options={[
            { value: "logit", label: "Logit Lens (Logit)" },
            { value: "probability", label: "Logit Lens (Probability)", disabled: !hasProbabilities },
          ]}
          onChange={setView}
        />
        <TraceDropdown
          label="Candidate tokens"
          value={candidateCount}
          options={candidateCounts.map((count) => ({ value: count, label: `Top ${count}` }))}
          onChange={setCandidateCount}
        />
      </div>
      <p className="chat-trace-title">Predictions for token #{selectedToken.index}: “{displayToken(selectedToken.tokenText)}”</p>
      <p className="chat-trace-caption">
        {hasProbabilities
          ? "The outlined token is the actual generated token when it appears in this layer’s displayed top candidates."
          : "This older trace has logits only. Regenerate this reply with Trace On to view exact probabilities."}
      </p>
      <div className="chat-trace-grid-region">
        <div className="chat-trace-grid-wrap" ref={gridWrapRef} onScroll={() => syncGridScroll("grid")}>
          <div className="chat-trace-grid" role="grid" aria-label="Logit Lens predictions" style={traceGridStyle}>
          <div className="chat-trace-grid-row chat-trace-grid-header-row" role="row">
            <div className="chat-trace-grid-layer-cell chat-trace-grid-header-cell" role="columnheader">Layer</div>
            {Array.from({ length: candidateCount }, (_, index) => (
              <div className="chat-trace-grid-rank-cell chat-trace-grid-header-cell" role="columnheader" key={index}>Rank {index + 1}</div>
            ))}
          </div>
          {layers.map((layer) => (
            <div className={`chat-trace-grid-row${layer.layer === selectedLayer ? " chat-trace-selected-row" : ""}`} role="row" key={layer.layer}>
              <button type="button" className="chat-trace-grid-layer-cell" onClick={() => setSelectedLayer(layer.layer)}>L{layer.layer}</button>
              {layer.candidates.slice(0, candidateCount).map((candidate) => (
                <button
                  type="button"
                  className={`chat-trace-grid-rank-cell${candidate.tokenId === selectedToken.tokenId ? " chat-trace-generated-token" : ""}`}
                  key={`${layer.layer}-${candidate.tokenId}`}
                  onClick={() => { setSelectedLayer(layer.layer); setSelectedCandidateId(candidate.tokenId); }}
                >
                  <span className="chat-trace-cell-token">{displayToken(candidate.tokenText)}</span>
                  <span className="chat-trace-cell-value">{metricValue(candidate, view)}</span>
                </button>
              ))}
            </div>
          ))}
          </div>
        </div>
        <div className="chat-trace-grid-scrollbar" ref={gridScrollbarRef} aria-label="Scroll Logit Lens ranks horizontally" onScroll={() => syncGridScroll("scrollbar")}>
          <div className="chat-trace-grid-scrollbar-spacer" style={{ width: gridScrollWidth }} />
        </div>
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
