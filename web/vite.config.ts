import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The build is served by FastAPI under /app/ (see src/counterpart/server.py).
export default defineConfig({
  plugins: [react()],
  base: "/app/",
  build: {
    outDir: "../src/counterpart/static/app",
    emptyOutDir: true,
  },
  // `npm run dev` (hot reload) proxies API calls to the FastAPI backend
  server: {
    proxy: { "/api": "http://127.0.0.1:8000" },
  },
});
