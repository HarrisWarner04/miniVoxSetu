import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// WHY: Vite is a modern build tool that's fast for development (uses native ES modules)
// and produces optimized bundles for production. We use it instead of Create React App
// because CRA is deprecated and Vite starts in <300ms.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // WHY: We proxy WebSocket connections to the backend so the frontend
    // doesn't need to know the backend URL — it just connects to /ws/chat
    // on the same origin. This avoids CORS issues with WebSockets.
    proxy: {
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
});
