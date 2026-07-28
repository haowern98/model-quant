import { useEffect, useRef, useState } from "react";
import type { ModelLoadConfig } from "./chat/chatTypes";

type ConfigField = {
  key: Exclude<keyof ModelLoadConfig, "thinking">;
  label: string;
  inputLabel: string;
  placeholder?: string;
  inputMode: "numeric" | "decimal";
};

const CONFIGURATION_FIELDS: readonly ConfigField[] = [
  { key: "seed", label: "Seed", inputLabel: "Load model seed", placeholder: "Random", inputMode: "numeric" },
  { key: "temperature", label: "Temperature", inputLabel: "Load model temperature", inputMode: "decimal" },
  { key: "topK", label: "Top K Sampling", inputLabel: "Load model top K sampling", inputMode: "numeric" },
  { key: "repeatPenalty", label: "Repeat Penalty", inputLabel: "Load model repeat penalty", inputMode: "decimal" },
  { key: "presencePenalty", label: "Presence Penalty", inputLabel: "Load model presence penalty", inputMode: "decimal" },
  { key: "topP", label: "Top P Sampling", inputLabel: "Load model top P sampling", inputMode: "decimal" },
  { key: "minP", label: "Min P Sampling", inputLabel: "Load model min P sampling", inputMode: "decimal" },
  { key: "contextWindow", label: "Context Window", inputLabel: "Load model context window", inputMode: "numeric" },
];

interface ModelLoadControlsProps {
  config: ModelLoadConfig;
  onConfigChange: (config: ModelLoadConfig) => void;
  hasModel: boolean;
  loaded: boolean;
  busy: boolean;
  onLoad: () => void;
  onUnload: () => void;
}

export function ModelLoadControls({ config, onConfigChange, loaded, busy, onLoad, onUnload }: ModelLoadControlsProps) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [thinkingMenuOpen, setThinkingMenuOpen] = useState(false);
  const menuRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!menuOpen) return;

    const closeMenu = () => {
      setMenuOpen(false);
      setThinkingMenuOpen(false);
    };
    const handlePointerDown = (event: PointerEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) closeMenu();
    };
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") closeMenu();
    };

    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  const updateField = (field: ConfigField, value: string) => {
    onConfigChange({ ...config, [field.key]: value });
  };

  const thinkingLabel = config.thinking === "on" ? "On" : "Off";

  return (
    <div className="editor-run-controls model-load-controls">
      <div
        ref={menuRef}
        className={`run-split-action ${menuOpen ? "open" : ""}`}
        role="group"
        aria-label="Model load controls"
        aria-busy={busy || undefined}
      >
        <button
          type="button"
          className="run-split-primary"
          aria-label={loaded ? "Unload model" : "Load model"}
          title={loaded ? "Unload model from Chat" : "Load model for Chat"}
          onClick={loaded ? onUnload : onLoad}
        >
          <span className={`codicon ${loaded ? "codicon-arrow-circle-down" : "codicon-arrow-circle-up"}`} aria-hidden="true" />
        </button>
        <button
          type="button"
          className="run-split-chevron"
          aria-label="Load model options"
          aria-expanded={menuOpen}
          aria-haspopup="dialog"
          title="Load configuration"
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span className="codicon codicon-chevron-down" aria-hidden="true" />
        </button>
        {menuOpen && (
          <section className="run-action-menu model-load-config-menu" role="dialog" aria-label="Model load configuration">
            <div className="run-menu-section-label">CONFIGURATION</div>
            <div className="model-load-config-fields">
              <label className="benchmark-info-row benchmark-input-row">
                <span>Seed</span>
                <input
                  aria-label="Load model seed"
                  className="benchmark-config-input"
                  value={config.seed}
                  placeholder="Random"
                  inputMode="numeric"
                  onChange={(event) => onConfigChange({ ...config, seed: event.currentTarget.value })}
                />
              </label>
              <div className="benchmark-info-row">
                <span>Thinking</span>
                <div className="benchmark-select-control">
                  <button
                    type="button"
                    className="benchmark-select-button"
                    aria-label={`Load model thinking ${thinkingLabel}`}
                    aria-expanded={thinkingMenuOpen}
                    aria-haspopup="listbox"
                    onClick={() => setThinkingMenuOpen((open) => !open)}
                  >
                    <span>{thinkingLabel}</span>
                    <span className="codicon codicon-chevron-down" aria-hidden="true" />
                  </button>
                  {thinkingMenuOpen && (
                    <div className="benchmark-select-menu" role="listbox" aria-label="Load model thinking options">
                      {[
                        { value: "off", label: "Off" },
                        { value: "on", label: "On" },
                      ].map((option) => (
                        <button
                          key={option.value}
                          type="button"
                          className="benchmark-select-option"
                          role="option"
                          aria-selected={config.thinking === option.value}
                          onMouseDown={(event) => event.preventDefault()}
                          onClick={() => {
                            onConfigChange({ ...config, thinking: option.value as ModelLoadConfig["thinking"] });
                            setThinkingMenuOpen(false);
                          }}
                        >
                          {option.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
              {CONFIGURATION_FIELDS.slice(1).map((field) => (
                <label className="benchmark-info-row benchmark-input-row" key={field.key}>
                  <span>{field.label}</span>
                  <input
                    aria-label={field.inputLabel}
                    className="benchmark-config-input"
                    value={config[field.key]}
                    placeholder={field.placeholder}
                    inputMode={field.inputMode}
                    onChange={(event) => updateField(field, event.currentTarget.value)}
                  />
                </label>
              ))}
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
