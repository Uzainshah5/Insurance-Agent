// App.jsx
// ---------------------------------------------------------------------------
// Root component.
// All conversation history and sessions are stored in SQLite via FastAPI.
// ---------------------------------------------------------------------------

import { useState, useRef, useEffect } from "react";
import { useMutation } from "@tanstack/react-query";

import Sidebar from "./components/Sidebar";
import ChatMessage from "./components/ChatMessage";
import ChatInput from "./components/ChatInput";
import {
  fetchSessions,
  createSession,
  fetchSessionMessages,
  deleteSession,
  sendChatMessage,
  checkServerHealth,
} from "./api/chatApi";
import "./App.css";

function App() {
  // -------------------------------------------------------------------------
  // 1. Sessions State (Loaded from SQLite backend)
  // -------------------------------------------------------------------------
  const [sessions, setSessions] = useState([]);
  const [currentSessionId, setCurrentSessionId] = useState(function () {
    return localStorage.getItem("rag_active_session_id") || null;
  });

  // -------------------------------------------------------------------------
  // 2. Active Messages State (Loaded from SQLite for current session)
  // -------------------------------------------------------------------------
  const [messages, setMessages] = useState([]);

  // Ref for auto-scrolling to the latest message
  const bottomRef = useRef(null);

  // -------------------------------------------------------------------------
  // 3. TanStack Query Mutation for sending messages
  // -------------------------------------------------------------------------
  const mutation = useMutation({
    mutationFn: function (payload) {
      // payload = { question, sessionId }
      return sendChatMessage(payload.question, payload.sessionId);
    },

    onSuccess: function (data) {
      // The backend saves this message directly into SQLite.
      // We append the assistant's response to local UI state.
      const assistantMessage = {
        role: "assistant",
        content: data.answer,
      };

      setMessages(function (previousMessages) {
        return [...previousMessages, assistantMessage];
      });

      // Refresh sessions list so any updated title is reflected in sidebar
      fetchSessions().then((updatedSessions) => {
        setSessions(updatedSessions);
      });
    },
  });

  // -------------------------------------------------------------------------
  // 4. Load sessions from SQLite on initial mount
  // -------------------------------------------------------------------------
  useEffect(() => {
    async function loadInitialData() {
      try {
        // Check server boot ID to detect server restart
        const health = await checkServerHealth();
        const lastBootId = localStorage.getItem("rag_server_boot_id");
        const serverRestarted = health.boot_id && health.boot_id !== lastBootId;

        if (health.boot_id) {
          localStorage.setItem("rag_server_boot_id", health.boot_id);
        }

        // Fetch all saved sessions from SQLite
        let loadedSessions = await fetchSessions();

        if (serverRestarted || loadedSessions.length === 0) {
          // Server restarted or no chats yet: start fresh session in SQLite
          const newSession = await createSession("New Chat");
          loadedSessions = [newSession, ...loadedSessions];
          setSessions(loadedSessions);
          setCurrentSessionId(newSession.id);
          setMessages([]);
          localStorage.setItem("rag_active_session_id", newSession.id);
          return;
        }

        setSessions(loadedSessions);

        // Determine which session to activate
        const savedActiveId = localStorage.getItem("rag_active_session_id");
        const sessionExists = loadedSessions.some((s) => s.id === savedActiveId);
        const activeId = sessionExists ? savedActiveId : loadedSessions[0].id;

        setCurrentSessionId(activeId);
        localStorage.setItem("rag_active_session_id", activeId);

        // Load messages for the active session from SQLite
        const sessionMsgs = await fetchSessionMessages(activeId);
        setMessages(sessionMsgs);
      } catch (err) {
        console.error("Failed to load initial data from SQLite backend:", err);
      }
    }

    loadInitialData();
  }, []);

  // -------------------------------------------------------------------------
  // Save active session ID so page refresh remembers which chat you were on
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (currentSessionId) {
      localStorage.setItem("rag_active_session_id", currentSessionId);
    }
  }, [currentSessionId]);

  // Auto-scroll whenever messages change or loading state changes
  useEffect(() => {
    if (bottomRef.current) {
      bottomRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages, mutation.isPending]);

  // -------------------------------------------------------------------------
  // 5. Session Actions (SQLite Backend Operations)
  // -------------------------------------------------------------------------

  // Start a new conversation
  async function handleNewChat() {
    // If top chat is already empty, just switch to it
    if (sessions.length > 0 && messages.length === 0 && sessions[0].title === "New Chat") {
      setCurrentSessionId(sessions[0].id);
      mutation.reset();
      return;
    }

    try {
      const newSession = await createSession("New Chat");
      setSessions((prev) => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
      setMessages([]);
      mutation.reset();
    } catch (err) {
      console.error("Failed to create new session in SQLite:", err);
    }
  }

  // Switch to an existing conversation
  async function handleSelectSession(sessionId) {
    if (sessionId === currentSessionId) return;

    setCurrentSessionId(sessionId);
    mutation.reset();

    try {
      // Fetch messages from SQLite for this selected session
      const sessionMsgs = await fetchSessionMessages(sessionId);
      setMessages(sessionMsgs);
    } catch (err) {
      console.error("Failed to fetch session messages from SQLite:", err);
    }
  }

  // Delete a conversation from SQLite
  async function handleDeleteSession(sessionIdToDelete) {
    try {
      await deleteSession(sessionIdToDelete);

      const remaining = sessions.filter((s) => s.id !== sessionIdToDelete);
      setSessions(remaining);

      // If we deleted the active chat, switch to the first remaining one
      if (currentSessionId === sessionIdToDelete) {
        if (remaining.length > 0) {
          const nextId = remaining[0].id;
          setCurrentSessionId(nextId);
          const nextMsgs = await fetchSessionMessages(nextId);
          setMessages(nextMsgs);
        } else {
          // If all sessions deleted, create a fresh one in SQLite
          const freshSession = await createSession("New Chat");
          setSessions([freshSession]);
          setCurrentSessionId(freshSession.id);
          setMessages([]);
        }
      }
    } catch (err) {
      console.error("Failed to delete session from SQLite:", err);
    }
  }

  // -------------------------------------------------------------------------
  // 6. Send Message Handler
  // -------------------------------------------------------------------------
  function handleSend(question) {
    if (!currentSessionId) return;

    // 1. Optimistically display user's message immediately on screen
    const userMessage = {
      role: "user",
      content: question,
    };
    setMessages((prev) => [...prev, userMessage]);

    // 2. If this was a "New Chat", update title preview in sidebar optimistically
    const currentSession = sessions.find((s) => s.id === currentSessionId);
    if (currentSession && currentSession.title === "New Chat") {
      const preview =
        question.slice(0, 25) + (question.length > 25 ? "..." : "");
      setSessions((prev) =>
        prev.map((s) => (s.id === currentSessionId ? { ...s, title: preview } : s))
      );
    }

    // 3. Fire mutation: the backend saves to SQLite and answers via RAG
    mutation.mutate({
      question: question,
      sessionId: currentSessionId,
    });
  }

  const currentSession = sessions.find((s) => s.id === currentSessionId);

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="layout-container">
      {/* Sidebar with all conversations stored in SQLite */}
      <Sidebar
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />

      {/* Main Chat Area */}
      <main className="main-chat">
        {/* Header */}
        <header className="app-header">
          <span className="app-title">🛡️ RAG Assistant</span>
          <span className="current-chat-title">{currentSession?.title}</span>
        </header>

        {/* Message list */}
        <div className="messages-container">
          {messages.length === 0 && !mutation.isPending && (
            <div className="empty-state">
              Ask me anything about Insurellm — products, employees, contracts and more.
            </div>
          )}

          {messages.map(function (message, index) {
            return <ChatMessage key={index} message={message} />;
          })}

          {/* Thinking indicator */}
          {mutation.isPending && (
            <div className="loading-row">
              <div className="loading-bubble">Thinking…</div>
            </div>
          )}

          {/* Invisible bottom anchor for auto-scroll */}
          <div ref={bottomRef} />
        </div>

        {/* Error banner */}
        {mutation.isError && (
          <div className="error-banner">
            ⚠️ Error:{" "}
            {mutation.error?.response?.data?.detail ||
              mutation.error?.message ||
              "Something went wrong. Please try again."}
          </div>
        )}

        {/* Input field fixed at bottom */}
        <div className="chat-input-wrapper">
          <ChatInput onSend={handleSend} isLoading={mutation.isPending} />
        </div>
      </main>
    </div>
  );
}

export default App;
