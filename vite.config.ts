import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
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
        index: resolve(__dirname, "index.html"),
        app: resolve(__dirname, "app/index.html"),
        report: resolve(__dirname, "report/index.html"),
      },
      preserveEntrySignatures: "exports-only",
      output: {
        preserveModules: true,
        preserveModulesRoot: __dirname,
        entryFileNames: "[name].js",
        chunkFileNames: "[name].js",
        assetFileNames: "assets/[name][extname]",
      },
    },
  },
});
