import { useState } from "react";
import {
  closeWindow,
  minimizeWindow,
  toggleMaximizeWindow,
} from "../lib/window-controls";

interface TitleBarProps {
  modelPath: string | null;
  onOpenModel: () => void;
}

function modelLabel(modelPath: string | null): string {
  if (!modelPath) return "Open GGUF...";
  return modelPath.split(/[\\/]/).pop() ?? modelPath;
}

export function TitleBar({ modelPath, onOpenModel }: TitleBarProps) {
  const [commandMenuOpen, setCommandMenuOpen] = useState(false);

  return (
    <header className="titlebar">
      <div className="titlebar-left">
        <div className="app-mark" aria-label="Model Surgery">
          <img src="/app-icon.png" alt="" aria-hidden="true" />
        </div>

        <div className="nav-buttons" aria-label="Navigation">
          <button type="button" aria-label="Back">
            <span className="codicon arrow-left" aria-hidden="true" />
          </button>
          <button type="button" aria-label="Forward">
            <span className="codicon arrow-right" aria-hidden="true" />
          </button>
        </div>
      </div>

      <div className="command-center-group">
        <button
          type="button"
          className="command-center"
          aria-label="Model Surgery command center"
          onClick={onOpenModel}
        >
          <span className="command-title">{modelLabel(modelPath)}</span>
        </button>
        <button
          type="button"
          className="command-center-dropdown"
          aria-label="Command center menu"
          aria-expanded={commandMenuOpen}
          onClick={() => setCommandMenuOpen((open) => !open)}
        >
          <span className="codicon codicon-chevron-down" aria-hidden="true" />
        </button>
        {commandMenuOpen && (
          <div className="command-center-menu" role="menu">
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setCommandMenuOpen(false);
                onOpenModel();
              }}
            >
              Open model GGUF...
            </button>
          </div>
        )}
      </div>

      <div className="titlebar-right">
        <div className="title-drag-fill" data-tauri-drag-region />

        <div className="title-actions" aria-label="Window controls">
          <button type="button" aria-label="Split layout">
            <span className="codicon split" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label="Minimize"
            onClick={() => void minimizeWindow()}
          >
            <span className="window-min" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label="Maximize"
            onClick={() => void toggleMaximizeWindow()}
          >
            <span className="window-max" aria-hidden="true" />
          </button>
          <button
            type="button"
            aria-label="Close"
            onClick={() => void closeWindow()}
          >
            <span className="window-close" aria-hidden="true" />
          </button>
        </div>
      </div>
    </header>
  );
}
