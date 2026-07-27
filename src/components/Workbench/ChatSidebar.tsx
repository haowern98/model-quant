import { useMemo, useState } from "react";
import type { ChatConversationSummary } from "./chat/chatTypes";

interface ChatSidebarProps {
  onNewChat: () => void;
  conversations: ChatConversationSummary[];
  onOpenChat: (id: string) => void;
}

export function ChatSidebar({ onNewChat, conversations, onOpenChat }: ChatSidebarProps) {
  const [query, setQuery] = useState("");
  const matchingConversations = useMemo(
    () => conversations.filter((conversation) => conversation.title.toLowerCase().includes(query.trim().toLowerCase())),
    [conversations, query],
  );

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
        {matchingConversations.map((conversation) => (
          <button
            key={conversation.id}
            type="button"
            className="chat-sidebar-conversation"
            title={conversation.title}
            onClick={() => onOpenChat(conversation.id)}
          >
            {conversation.title}
          </button>
        ))}
      </div>
    </aside>
  );
}
