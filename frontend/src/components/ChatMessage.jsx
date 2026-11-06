// ChatMessage.jsx
// ---------------------------------------------------------------------------
// Renders a single chat message bubble.
// User messages appear on the right; assistant messages on the left.
// Automatically formats text wrapped in ** or *** as bold <strong> elements.
// ---------------------------------------------------------------------------

function formatBoldText(content) {
  if (!content) return "";

  // Regular expression to find text wrapped in 2 or 3 asterisks:
  // e.g. **Markellm** or ***Markellm***
  const boldPattern = /\*{2,3}([^*]+?)\*{2,3}/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = boldPattern.exec(content)) !== null) {
    // 1. Add any plain text that appeared before this match
    if (match.index > lastIndex) {
      parts.push(content.slice(lastIndex, match.index));
    }

    // 2. Add the inner matched text wrapped in <strong> (bold) without the asterisks
    parts.push(<strong key={match.index}>{match[1]}</strong>);

    // 3. Move the pointer forward past this match
    lastIndex = boldPattern.lastIndex;
  }

  // 4. Add any remaining plain text after the final match
  if (lastIndex < content.length) {
    parts.push(content.slice(lastIndex));
  }

  return parts.length > 0 ? parts : content;
}

function ChatMessage({ message }) {
  // Decide which CSS class to apply based on who sent the message.
  const isUser = message.role === "user";
  const bubbleClass = isUser ? "bubble bubble-user" : "bubble bubble-assistant";
  const rowClass = isUser
    ? "message-row message-row-user"
    : "message-row message-row-assistant";

  return (
    <div className={rowClass}>
      {/* Show label above each bubble */}
      <div className="message-label">{isUser ? "You" : "RAG Assistant"}</div>

      {/* Bubble text with bold styling applied */}
      <div className={bubbleClass}>{formatBoldText(message.content)}</div>
    </div>
  );
}

export default ChatMessage;
