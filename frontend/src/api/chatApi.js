import axios from "axios";

// ---------------------------------------------------------------------------
// Axios instance
// ---------------------------------------------------------------------------
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || "http://localhost:8000",
  headers: {
    "Content-Type": "application/json",
  },
});

// ---------------------------------------------------------------------------
// SQLite Session Endpoints
// ---------------------------------------------------------------------------

// Get all saved sessions from SQLite
export async function fetchSessions() {
  const response = await apiClient.get("/api/sessions");
  return response.data;
}

// Create a new session in SQLite
export async function createSession(title = "New Chat") {
  const response = await apiClient.post("/api/sessions", { title });
  return response.data;
}

// Get all messages for a specific session from SQLite
export async function fetchSessionMessages(sessionId) {
  const response = await apiClient.get(`/api/sessions/${sessionId}/messages`);
  return response.data;
}

// Delete a session and its messages from SQLite
export async function deleteSession(sessionId) {
  const response = await apiClient.delete(`/api/sessions/${sessionId}`);
  return response.data;
}

// ---------------------------------------------------------------------------
// Send Chat Message (saved directly into SQLite by the backend)
// ---------------------------------------------------------------------------
export async function sendChatMessage(question, sessionId) {
  const response = await apiClient.post("/api/chat", {
    question: question,
    session_id: sessionId,
  });
  return response.data;
}

// ---------------------------------------------------------------------------
// Server Health Check
// ---------------------------------------------------------------------------
export async function checkServerHealth() {
  const response = await apiClient.get("/health");
  return response.data;
}
