// ChatInput.jsx
// ---------------------------------------------------------------------------
// The input bar fixed at the bottom of the screen.
// Contains a text field and a Send button.
// When the user submits, it calls onSend(inputText) and clears the field.
// ---------------------------------------------------------------------------

import { useState } from "react";

function ChatInput({ onSend, isLoading }) {
  // Local state: what the user has typed so far
  const [inputText, setInputText] = useState("");

  // Called when the user clicks Send or presses Enter
  function handleSubmit(event) {
    // Prevent the default browser form submission which would reload the page
    event.preventDefault();

    // Don't send if the field is empty or if we're already waiting for a reply
    const trimmed = inputText.trim();
    if (!trimmed || isLoading) {
      return;
    }

    // Pass the question up to the parent (App.jsx) and clear the input
    onSend(trimmed);
    setInputText("");
  }

  // Allow pressing Enter (without Shift) to submit the form
  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      handleSubmit(event);
    }
  }

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        className="chat-input-field"
        placeholder="Ask a question about Insurellm..."
        value={inputText}
        onChange={(event) => setInputText(event.target.value)}
        onKeyDown={handleKeyDown}
        // Disable while waiting so the user can't send multiple requests at once
        disabled={isLoading}
      />
      <button
        type="submit"
        className="send-button"
        disabled={isLoading || !inputText.trim()}
      >
        {isLoading ? "..." : "Send"}
      </button>
    </form>
  );
}

export default ChatInput;

