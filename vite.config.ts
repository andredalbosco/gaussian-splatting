import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "dist",
    rollupOptions: {
      input: {
        app: resolve(__dirname, "app/index.html"),
        report: resolve(__dirname, "report/index.html"),
      },
    },
  },
});
