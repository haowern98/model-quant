type ActivityId = "gguf" | "chat" | "server" | "settings";

interface ActivityItem {
  id: ActivityId;
  label: string;
  icon: string;
}

const TOP_ITEMS: ActivityItem[] = [
  { id: "gguf", label: "Explorer view", icon: "files" },
  { id: "chat", label: "Chat with model", icon: "comment-discussion" },
  { id: "server", label: "Server mode", icon: "server-process" },
];

const BOTTOM_ITEMS: ActivityItem[] = [
  { id: "settings", label: "Settings", icon: "settings-gear" },
];

function ActivityIcon({ icon }: { icon: string }) {
  return <span className={`activity-icon codicon codicon-${icon}`} aria-hidden="true" />;
}

export function ActivityBar() {
  return (
    <aside className="activity-bar" aria-label="Primary navigation">
      <div className="activity-group">
        {TOP_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`activity-button ${item.id === "gguf" ? "active" : ""}`}
            aria-label={item.label}
          >
            <ActivityIcon icon={item.icon} />
          </button>
        ))}
      </div>
      <div className="activity-group">
        {BOTTOM_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className="activity-button"
            aria-label={item.label}
          >
            <ActivityIcon icon={item.icon} />
          </button>
        ))}
      </div>
    </aside>
  );
}
