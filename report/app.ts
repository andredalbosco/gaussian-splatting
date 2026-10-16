import "../app/main";

const overlay = document.querySelector<HTMLDivElement>(".overlay");
const appFrame = document.querySelector<HTMLDivElement>(".app-frame");

if (overlay && appFrame) {
  const stats = overlay.firstElementChild as HTMLElement | null;
  const pane = overlay.lastElementChild as HTMLElement | null;

  appFrame.style.position = "relative";

  if (stats) {
    stats.style.position = "absolute";
    stats.style.top = "12px";
    stats.style.left = "12px";
    stats.style.zIndex = "1";
    stats.style.pointerEvents = "none";
    appFrame.appendChild(stats);
  }

  if (pane) {
    overlay.replaceChildren(pane);
  }

  appFrame.after(overlay);
}
