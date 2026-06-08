import type { ReactNode } from "react";

export function ExplorerSectionHeader({
  label,
  expanded,
  ariaLabel,
  onClick,
  action,
}: {
  label: string;
  expanded: boolean;
  ariaLabel?: string;
  onClick: () => void;
  action?: ReactNode;
}) {
  return (
    <div className="explorer-section-header explorer-model-header">
      <button
        type="button"
        className="explorer-section-toggle"
        aria-label={ariaLabel ?? label}
        onClick={onClick}
      >
        <span className={`tree-chevron ${expanded ? "expanded" : ""}`} />
        <span>{label}</span>
      </button>
      {action ?? <span className="tree-action-spacer" aria-hidden="true" />}
    </div>
  );
}

export function ExplorerTreeRow({
  label,
  right,
  expanded,
  active = false,
  ariaLabel,
  onClick,
  onToggle,
}: {
  label: string;
  right?: ReactNode;
  expanded?: boolean;
  active?: boolean;
  ariaLabel?: string;
  onClick?: () => void;
  onToggle?: () => void;
}) {
  return (
    <button
      type="button"
      className={`tree-row layer-row ${active ? "active" : ""}`}
      aria-label={ariaLabel ?? `${label}${right !== undefined ? ` ${String(right)}` : ""}`}
      onClick={onClick}
    >
      <span
        className={`tree-chevron ${expanded ? "expanded" : ""}`}
        onClick={(event) => {
          if (!onToggle) return;
          event.stopPropagation();
          onToggle();
        }}
      />
      <span className="tree-folder-icon" aria-hidden="true" />
      <span className="tree-label">{label}</span>
      <span className="tree-count">{right}</span>
    </button>
  );
}
