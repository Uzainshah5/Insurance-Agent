// Sidebar.jsx
// ---------------------------------------------------------------------------
// Displays a list of past conversation sessions and a "+ New Chat" button.
// Beginner-friendly component: just receives props and renders a list.
// ---------------------------------------------------------------------------

function Sidebar({
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
}) {
  return (
    <aside className="sidebar">
      {/* Top button to start a fresh conversation */}
      <div className="sidebar-header">
        <button className="sidebar-new-button" onClick={onNewChat}>
          + New Chat
        </button>
      </div>

      {/* Label for previous conversations */}
      <div className="sidebar-section-title">Conversations</div>

      {/* List of all sessions */}
      <div className="sidebar-list">
        {sessions.map((session) => {
          const isActive = session.id === currentSessionId;

          return (
            <div
              key={session.id}
              className={`sidebar-item ${isActive ? "sidebar-item-active" : ""}`}
              onClick={() => onSelectSession(session.id)}
            >
              {/* Session title with a speech bubble icon */}
              <span className="sidebar-item-title" title={session.title}>
                💬 {session.title}
              </span>

              {/* Delete button (only show when there's more than 1 chat) */}
              {sessions.length > 1 && (
                <button
                  className="sidebar-item-delete"
                  title="Delete conversation"
                  onClick={(event) => {
                    // Prevent clicking the delete button from triggering onSelectSession
                    event.stopPropagation();
                    onDeleteSession(session.id);
                  }}
                >
                  ✕
                </button>
              )}
            </div>
          );
        })}
      </div>
    </aside>
  );
}

export default Sidebar;

