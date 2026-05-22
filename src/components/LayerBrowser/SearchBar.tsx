interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
}

export function SearchBar({ value, onChange }: SearchBarProps) {
  return (
    <div className="px-3 pt-3 pb-2">
      <input
        type="text"
        value={value}
        onChange={e => onChange(e.target.value)}
        placeholder="Filter layers..."
        className="w-full bg-bg-surface-alt border border-border-default rounded px-2 py-1 text-sm text-text-primary
                   placeholder:text-text-muted focus:outline-none focus:border-accent-copper font-mono"
      />
    </div>
  );
}
