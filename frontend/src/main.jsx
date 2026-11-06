// main.jsx
// ---------------------------------------------------------------------------
// Entry point.  Sets up TanStack Query and renders the App.
// ---------------------------------------------------------------------------

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import App from "./App.jsx";

// ---------------------------------------------------------------------------
// QueryClient
// ---------------------------------------------------------------------------
// QueryClient is TanStack Query's central store.
// We configure it once here and pass it down via QueryClientProvider so that
// every component in the tree can use useQuery / useMutation.
//
// retry: 1  — if a request fails, try once more before showing an error.
// ---------------------------------------------------------------------------
const queryClient = new QueryClient({
  defaultOptions: {
    mutations: {
      retry: 1,
    },
  },
});

createRoot(document.getElementById("root")).render(
  <StrictMode>
    {/* QueryClientProvider makes queryClient available to all child components */}
    <QueryClientProvider client={queryClient}>
      <App />
    </QueryClientProvider>
  </StrictMode>
);
