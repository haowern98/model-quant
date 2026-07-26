export function ChatSidebar() {
  return (
    <aside className="explorer-panel" aria-label="Chat">
      <div className="explorer-title">
        <span>CHAT</span>
      </div>
      <div className="explorer-section chat-sidebar-divider" aria-hidden="true" />
      <div className="chat-sidebar-search">
        <span className="codicon codicon-search" aria-hidden="true" />
        <input type="search" aria-label="Search chats" placeholder="Search chats..." />
      </div>
    </aside>
  );
}
