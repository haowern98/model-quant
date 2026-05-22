export function TitleBar() {
  return (
    <div
      className="h-10 bg-bg-surface border-b border-border-default flex items-center px-4 select-none"
      data-tauri-drag-region
    >
      <h1 className="font-heading text-sm font-semibold text-accent-copper tracking-widest uppercase">
        Model Surgery
      </h1>
    </div>
  );
}
