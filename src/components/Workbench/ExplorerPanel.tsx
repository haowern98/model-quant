import { useEffect, useMemo, useRef, useState } from "react";
import {
  QUANT_TYPES,
  type AssignPattern,
  type QuantType,
  type TensorInfo,
} from "../../types";
import { formatTensorName } from "../../lib/format";
import { ExplorerSectionHeader, ExplorerTreeRow } from "./ExplorerTree";

type ExplorerSectionId = "gguf" | "mmproj" | "lora";

const BULK_PATTERNS: { value: AssignPattern; label: string; aria: string }[] = [
  { value: "all_attn", label: "All Attention", aria: "All Attention target" },
  { value: "all_ffn", label: "All FFN", aria: "All FFN target" },
  { value: "all_embeddings", label: "All Embeddings", aria: "All Embeddings target" },
  { value: "all", label: "Entire Model", aria: "Entire Model target" },
];

interface ExplorerPanelProps {
  modelPath: string | null;
  tensors: TensorInfo[];
  activeLayerIndex: number | null;
  expandedLayers: Set<number>;
  running: boolean;
  onOpenLayer: (layerIndex: number) => void;
  onOpenTensorValues: (tensor: TensorInfo, layerLabel: string) => void;
  onOpenModel: () => void;
  onToggleLayer: (layerIndex: number) => void;
  onAssignByPattern: (pattern: AssignPattern, quantType: QuantType) => void;
  onSaveRecipe: () => void;
  onLoadRecipe: () => void;
  onExport: () => void;
}

function basename(path: string | null): string {
  if (!path) return "GGUF";
  return path.split(/[\\/]/).pop() ?? path;
}

function sectionLabel(layerIndex: number, layerTensors: TensorInfo[]): string {
  if (layerIndex < 0) return "Global tensors";
  const parts = layerTensors[0]?.name.split(".").filter(Boolean) ?? [];
  const numberIndex = parts.findIndex((part) => /^\d+$/.test(part));
  if (numberIndex > 0) return parts.slice(0, numberIndex + 1).join(".");
  return `Layer ${layerIndex}`;
}

export function ExplorerPanel({
  modelPath,
  tensors,
  activeLayerIndex,
  expandedLayers,
  running,
  onOpenLayer,
  onOpenTensorValues,
  onOpenModel,
  onToggleLayer,
  onAssignByPattern,
  onSaveRecipe,
  onLoadRecipe,
  onExport,
}: ExplorerPanelProps) {
  const [sections, setSections] = useState<Record<ExplorerSectionId, boolean>>({
    gguf: true,
    mmproj: false,
    lora: false,
  });
  const [actionsOpen, setActionsOpen] = useState(false);
  const sectionBodyRef = useRef<HTMLDivElement>(null);
  const layerGroupRefs = useRef(new Map<number, HTMLDivElement>());
  const [stickyLayerIndices, setStickyLayerIndices] = useState<Set<number>>(() => new Set());

  const groups = useMemo(() => {
    const next = new Map<number, TensorInfo[]>();
    for (const tensor of tensors) {
      const group = next.get(tensor.layerIndex) ?? [];
      group.push(tensor);
      next.set(tensor.layerIndex, group);
    }
    return [...next.entries()].sort(([a], [b]) => a - b);
  }, [tensors]);

  const toggleSection = (section: ExplorerSectionId) => {
    setSections((current) => ({ ...current, [section]: !current[section] }));
  };

  useEffect(() => {
    const scrollBody = sectionBodyRef.current;
    if (!scrollBody) return;

    let frame = 0;
    const updateStickyLayers = () => {
      frame = 0;
      const scrollTop = scrollBody.getBoundingClientRect().top;
      const next = new Set<number>();

      for (const [layerIndex, group] of layerGroupRefs.current) {
        if (!expandedLayers.has(layerIndex)) continue;
        const header = group.firstElementChild;
        if (!header) continue;
        const headerRect = header.getBoundingClientRect();
        if (headerRect.top <= scrollTop && group.getBoundingClientRect().bottom > scrollTop + headerRect.height) {
          next.add(layerIndex);
        }
      }

      setStickyLayerIndices((current) => {
        if (current.size === next.size && [...current].every((layerIndex) => next.has(layerIndex))) return current;
        return next;
      });
    };
    const scheduleUpdate = () => {
      if (frame) return;
      frame = window.requestAnimationFrame(updateStickyLayers);
    };

    scheduleUpdate();
    scrollBody.addEventListener("scroll", scheduleUpdate, { passive: true });
    window.addEventListener("resize", scheduleUpdate);
    return () => {
      if (frame) window.cancelAnimationFrame(frame);
      scrollBody.removeEventListener("scroll", scheduleUpdate);
      window.removeEventListener("resize", scheduleUpdate);
    };
  }, [expandedLayers, groups]);

  const handleBulkAssign = (pattern: AssignPattern, value: string) => {
    if (!value) return;
    onAssignByPattern(pattern, value as QuantType);
    setActionsOpen(false);
  };

  return (
    <aside className="explorer-panel" aria-label="Explorer">
      <div className="explorer-title">
        <span>MODEL EXPLORER</span>
        <button type="button" aria-label="Explorer actions">...</button>
      </div>

      <section className="explorer-section">
        <ExplorerSectionHeader
          label={basename(modelPath)}
          ariaLabel={basename(modelPath)}
          expanded={sections.gguf}
          onClick={() => toggleSection("gguf")}
          action={
            modelPath ? (
            <button
              type="button"
              className="tree-action-button"
              aria-label="Model actions"
              onClick={() => setActionsOpen((value) => !value)}
            >
              ...
            </button>
            ) : undefined
          }
        />

        {sections.gguf && (
          <div className="explorer-section-body" ref={sectionBodyRef}>
            {actionsOpen && modelPath && (
              <div className="model-actions-popover">
                <div className="model-actions-header">Recipe Actions</div>
                <div className="model-action-buttons">
                  <button type="button" onClick={onSaveRecipe} disabled={running}>
                    Save Recipe
                  </button>
                  <button type="button" onClick={onLoadRecipe} disabled={running}>
                    Load Recipe
                  </button>
                  <button type="button" onClick={onExport} disabled={running}>
                    Export GGUF
                  </button>
                </div>
                <div className="model-actions-header">Bulk Assign</div>
                {BULK_PATTERNS.map((pattern) => (
                  <label key={pattern.value} className="bulk-action-row">
                    <span>{pattern.label}</span>
                    <select
                      aria-label={pattern.aria}
                      disabled={running}
                      defaultValue=""
                      onChange={(event) => {
                        handleBulkAssign(pattern.value, event.target.value);
                        event.currentTarget.value = "";
                      }}
                    >
                      <option value="">Apply...</option>
                      {QUANT_TYPES.map((quant) => (
                        <option key={quant.value} value={quant.value}>
                          {quant.label}
                        </option>
                      ))}
                    </select>
                  </label>
                ))}
              </div>
            )}

            {!modelPath && (
              <div className="future-section-empty">
                <button type="button" onClick={onOpenModel}>
                  Add model GGUF...
                </button>
              </div>
            )}

            {modelPath &&
              groups.map(([layerIndex, layerTensors]) => {
                const expanded = expandedLayers.has(layerIndex);
                const active = activeLayerIndex === layerIndex;
                const label = sectionLabel(layerIndex, layerTensors);
                return (
                  <div
                    key={layerIndex}
                    className={`explorer-layer-group ${expanded ? "expanded" : ""} ${
                      stickyLayerIndices.has(layerIndex) ? "sticky-shadow" : ""
                    }`}
                    ref={(node) => {
                      if (node) layerGroupRefs.current.set(layerIndex, node);
                      else layerGroupRefs.current.delete(layerIndex);
                    }}
                  >
                    <ExplorerTreeRow
                      label={label}
                      right={layerTensors.length}
                      expanded={expanded}
                      active={active}
                      aria-label={`${label} ${layerTensors.length}`}
                      onClick={() => {
                        if (expanded) onToggleLayer(layerIndex);
                        else onOpenLayer(layerIndex);
                      }}
                      onToggle={() => onToggleLayer(layerIndex)}
                    />
                    {expanded &&
                      layerTensors.map((tensor) => (
                        <button
                          type="button"
                          key={tensor.name}
                          className="tensor-child-row"
                          title={tensor.name}
                          onClick={() => onOpenTensorValues(tensor, label)}
                        >
                          {formatTensorName(tensor.name)}
                        </button>
                      ))}
                  </div>
                );
              })}
          </div>
        )}
      </section>

      <FutureSection
        id="mmproj"
        label="MMPROJ"
        expanded={sections.mmproj}
        onToggle={toggleSection}
        emptyLabel="Add projector..."
      />
      <FutureSection
        id="lora"
        label="LORA ADAPTER"
        expanded={sections.lora}
        onToggle={toggleSection}
        emptyLabel="Add adapter..."
      />
    </aside>
  );
}

function FutureSection({
  id,
  label,
  expanded,
  onToggle,
  emptyLabel,
}: {
  id: ExplorerSectionId;
  label: string;
  expanded: boolean;
  onToggle: (id: ExplorerSectionId) => void;
  emptyLabel: string;
}) {
  return (
    <section className="explorer-section">
      <button
        type="button"
        className="explorer-section-header"
        aria-label={label}
        onClick={() => onToggle(id)}
      >
        <span className={`tree-chevron ${expanded ? "expanded" : ""}`} />
        <span>{label}</span>
      </button>
      {expanded && (
        <div className="future-section-empty">
          <button type="button">{emptyLabel}</button>
        </div>
      )}
    </section>
  );
}
