export type ActivityId = "gguf" | "chat" | "testing" | "server" | "settings";

interface ActivityItem {
  id: ActivityId;
  label: string;
  icon: string;
}

const TOP_ITEMS: ActivityItem[] = [
  { id: "gguf", label: "Explorer view", icon: "files" },
  { id: "chat", label: "Chat with model", icon: "comment-discussion" },
  { id: "testing", label: "Testing", icon: "beaker" },
  { id: "server", label: "Server mode", icon: "server-process" },
];

const BOTTOM_ITEMS: ActivityItem[] = [
  { id: "settings", label: "Settings", icon: "settings-gear" },
];

function isSelectableActivity(activity: ActivityId): activity is "gguf" | "testing" {
  return activity === "gguf" || activity === "testing";
}

function ActivityIcon({ icon }: { icon: string }) {
  return <span className={`activity-icon codicon codicon-${icon}`} aria-hidden="true" />;
}

interface ActivityBarProps {
  activeActivity: ActivityId;
  panelVisible: boolean;
  onSelectActivity: (activity: ActivityId) => void;
}

export function ActivityBar({ activeActivity, panelVisible, onSelectActivity }: ActivityBarProps) {
  return (
    <aside className="activity-bar" aria-label="Primary navigation">
      <div className="activity-group">
        {TOP_ITEMS.map((item) => (
          <button
            key={item.id}
            type="button"
            className={`activity-button ${
              item.id === activeActivity && panelVisible && isSelectableActivity(item.id)
                ? "active"
                : ""
            }`}
            aria-label={item.label}
            aria-pressed={
              isSelectableActivity(item.id) ? item.id === activeActivity && panelVisible : undefined
            }
            onClick={isSelectableActivity(item.id) ? () => onSelectActivity(item.id) : undefined}
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
