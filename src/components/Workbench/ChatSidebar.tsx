import { useEffect, useMemo, useRef, useState } from "react";
import type { ChatConversationSummary } from "./chat/chatTypes";

interface ChatSidebarProps {
  onNewChat: () => void;
  conversations: ChatConversationSummary[];
  onOpenChat: (id: string) => void;
  onDeleteChat: (id: string) => Promise<void>;
  activeConversationId: string | null;
  deletionBlockedConversationIds: ReadonlySet<string>;
}

export function ChatSidebar({
  onNewChat,
  conversations,
  onOpenChat,
  onDeleteChat,
  activeConversationId,
  deletionBlockedConversationIds,
}: ChatSidebarProps) {
  const [query, setQuery] = useState("");
  const [openMenuId, setOpenMenuId] = useState<string | null>(null);
  const actionsRowRef = useRef<HTMLDivElement>(null);
  const matchingConversations = useMemo(
    () => conversations.filter((conversation) => conversation.title.toLowerCase().includes(query.trim().toLowerCase())),
    [conversations, query],
  );

  useEffect(() => {
    if (!openMenuId) return;
    const closeMenu = (event: MouseEvent) => {
      if (!actionsRowRef.current?.contains(event.target as Node)) setOpenMenuId(null);
    };
    window.addEventListener("pointerdown", closeMenu);
    return () => window.removeEventListener("pointerdown", closeMenu);
  }, [openMenuId]);

  const deleteConversation = async (conversation: ChatConversationSummary) => {
    if (deletionBlockedConversationIds.has(conversation.id)) return;
    if (!window.confirm("Delete this chat and all of its saved traces? This can't be undone.")) return;
    setOpenMenuId(null);
    await onDeleteChat(conversation.id);
  };

  return (
    <aside className="explorer-panel" aria-label="Chat">
      <div className="explorer-title">
        <span>CHAT</span>
      </div>
      <div className="explorer-section chat-sidebar-divider" aria-hidden="true" />
      <div className="chat-sidebar-search">
        <span className="codicon codicon-search" aria-hidden="true" />
        <input
          type="search"
          aria-label="Search chats"
          placeholder="Search chats..."
          value={query}
          onChange={(event) => setQuery(event.currentTarget.value)}
        />
      </div>
      <button type="button" className="chat-sidebar-new-chat" onClick={onNewChat}>
        <span className="codicon codicon-edit" aria-hidden="true" />
        <span>New chat</span>
      </button>
      <div className="chat-sidebar-conversations" aria-label="Saved chats">
        {matchingConversations.map((conversation) => {
          const deletionBlocked = deletionBlockedConversationIds.has(conversation.id);
          const menuOpen = openMenuId === conversation.id;
          return (
            <div
              key={conversation.id}
              ref={menuOpen ? actionsRowRef : undefined}
              className={`chat-sidebar-conversation-row${activeConversationId === conversation.id ? " is-active" : ""}${menuOpen ? " has-open-menu" : ""}`}
            >
              <button
                type="button"
                className="chat-sidebar-conversation"
                title={conversation.title}
                onClick={() => {
                  setOpenMenuId(null);
                  onOpenChat(conversation.id);
                }}
              >
                <span>{conversation.title}</span>
              </button>
              <button
                type="button"
                className="chat-sidebar-conversation-actions"
                aria-label={`Chat actions for ${conversation.title}`}
                aria-haspopup="menu"
                aria-expanded={menuOpen}
                onClick={() => setOpenMenuId((current) => current === conversation.id ? null : conversation.id)}
              >
                <span className="codicon codicon-ellipsis" aria-hidden="true" />
              </button>
              {menuOpen ? (
                <div className="chat-sidebar-action-menu" role="menu">
                  <button
                    type="button"
                    role="menuitem"
                    disabled={deletionBlocked}
                    title={deletionBlocked ? "Wait for this chat to finish before deleting it." : undefined}
                    onClick={() => void deleteConversation(conversation)}
                  >
                    Delete chat
                  </button>
                </div>
              ) : null}
            </div>
          );
        })}
      </div>
    </aside>
  );
}
