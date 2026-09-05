/* Entry point. Mounts the shell once, registers the routes, boots the session.
 *
 * Pages are loaded with dynamic `import()` so the first paint only costs the
 * shell plus the landing page. There is no bundler, so this is a real network
 * win rather than a bookkeeping exercise: nine page modules would otherwise be
 * nine blocking requests before anything renders.
 */

import { el, mount } from "./dom.js";
import { route, fallback, start } from "./router.js";
import { a11yLayer, sidebar, tabbar, topbar } from "./components/shell.js";
import { applyTheme, bootSession, ensureHealth, restoreMeal } from "./store.js";
import { toastWarn } from "./toast.js";
import { tooltips } from "./tooltip.js";

/* ------------------------------------------------------------------- Shell */

applyTheme();
tooltips();

const outlet = el("main", { class: "main", id: "main", role: "main", tabindex: "-1" });

mount(document.getElementById("app"), ...a11yLayer(), sidebar(), topbar(), outlet, tabbar());

/* ------------------------------------------------------------------ Routes */

/** Wrap a page module in a dynamic import.
 *
 *  This awaits the import but *not* the page's data — pages return their node
 *  immediately and fill in their own skeletons. That ordering matters: the
 *  router moves focus to the new view's heading as soon as the loader resolves
 *  (router.js:96), so a page that resolved only after its fetch would leave
 *  keyboard focus stranded on the old view for the duration of the request.
 */
const lazy = (importer) => async (context) => {
  try {
    const module = await importer();
    return await module.default(context);
  } catch (error) {
    console.error("[app] page failed to load", error);
    return loadFailure(error);
  }
};

function loadFailure(error) {
  return el(
    "div",
    { class: "page" },
    el(
      "div",
      { class: "empty" },
      el("h1", { text: "This page didn't load" }),
      el("p", { class: "muted", text: error?.message || "Something went wrong on the way here." }),
      el("button", { class: "btn btn--primary", text: "Reload", onclick: () => location.reload() })
    )
  );
}

route("/", lazy(() => import("./pages/home.js")));
route("/analyzing", lazy(() => import("./pages/analyzing.js")));
route("/review", lazy(() => import("./pages/review.js")));
route("/results", lazy(() => import("./pages/results.js")));
route("/meal/:id", lazy(() => import("./pages/results.js")));
route("/today", lazy(() => import("./pages/today.js")));
route("/history", lazy(() => import("./pages/history.js")));
route("/method", lazy(() => import("./pages/method.js")));
route("/settings", lazy(() => import("./pages/settings.js")));
route("/auth", lazy(() => import("./pages/auth.js")));
fallback(lazy(() => import("./pages/notfound.js")));

/* -------------------------------------------------------------------- Boot */

/* The session must resolve before the first page renders, or a guest lands on
 * Today and sees an auth error instead of an empty state. Health is
 * fire-and-forget: it only drives the engine badge in the top bar. */

restoreMeal();

ensureHealth()
  .then((health) => {
    if (health?.status && health.status !== "ok") {
      toastWarn("The analysis models are still starting up — an upload may fail for a moment.", {
        title: "Warming up",
      });
    }
  })
  .catch(() => {
    /* Badge stays hidden. Every real action reports its own failure. */
  });

bootSession()
  .catch(() => null)
  .finally(() => {
    document.documentElement.dataset.booted = "true";
    start(outlet);
  });

window.addEventListener("unhandledrejection", (event) => {
  if (event.reason?.name === "AbortError") return;
  console.error("[app] unhandled rejection", event.reason);
});
