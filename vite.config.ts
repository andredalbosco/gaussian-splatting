import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "dist",
    minify: false,
    cssMinify: false,
    sourcemap: true,
    target: "esnext",
    modulePreload: {
      polyfill: false,
    },
    rollupOptions: {
      input: {
        app: resolve(__dirname, "app/index.html"),
        report: resolve(__dirname, "report/index.html"),
      },
      output: {
        entryFileNames: ({ name }) => `${name}/index.js`,
        chunkFileNames: "assets/[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
});
