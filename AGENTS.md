# AGENTS

- Use Deno-backed tooling for this repo:
  - `deno run -A npm:vite`
  - `deno run -A npm:vite build`
  - `deno run -A npm:typescript@5.9.3/tsc --noEmit`
- This is a multi-page Vite app with independent entries at `app/index.html` and `report/index.html`.
- Keep page code isolated: `app/*` is for the app page, `report/*` is for the report page.
- `vite.config.ts` is included in TypeScript checks. If it uses Node APIs like `node:path` or `__dirname`, add proper Node typing/config support instead of bypassing the error.
