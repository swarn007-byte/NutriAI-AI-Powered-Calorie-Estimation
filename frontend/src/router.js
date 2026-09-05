/* History API router. Path-based rather than hash-based because the FastAPI
 * catch-all in main.py:689 already serves index.html for every unmatched path,
 * so deep links work on a hard refresh and the URLs stay clean.
 *
 * Routes are matched in registration order against a `/segment/:param` pattern.
 * There are eight of them, so a compiled trie would be over-engineering.
 */

const routes = [];
let notFound = null;
let outlet = null;
let current = null;
let renderToken = 0;

export function route(pattern, load) {
  routes.push({ pattern, segments: pattern.split("/").filter(Boolean), load });
}

export function fallback(load) {
  notFound = load;
}

function match(path) {
  const parts = path.split("/").filter(Boolean);
  for (const entry of routes) {
    if (entry.segments.length !== parts.length) continue;
    const params = {};
    let ok = true;
    for (let i = 0; i < entry.segments.length; i += 1) {
      const segment = entry.segments[i];
      if (segment.startsWith(":")) params[segment.slice(1)] = decodeURIComponent(parts[i]);
      else if (segment !== parts[i]) {
        ok = false;
        break;
      }
    }
    if (ok) return { entry, params };
  }
  return null;
}

export function navigate(to, { replace = false, state: histState = null } = {}) {
  const url = new URL(to, location.origin);
  const same = url.pathname === location.pathname && url.search === location.search;
  if (same) return render();
  if (replace) history.replaceState(histState, "", url);
  else history.pushState(histState, "", url);
  return render();
}

export const currentPath = () => location.pathname;

/** Scroll restoration: back/forward returns to the stored offset, a fresh
 *  navigation goes to the top. Browsers get this wrong for replaced content. */
const offsets = new Map();

function rememberOffset() {
  if (current) offsets.set(current, window.scrollY);
}

async function render() {
  const path = location.pathname;
  const found = match(path);
  const loader = found ? found.entry.load : notFound;
  if (!loader || !outlet) return;

  const token = ++renderToken;
  const params = found ? found.params : {};
  const query = Object.fromEntries(new URLSearchParams(location.search));

  let view;
  try {
    view = await loader({ params, query, path });
  } catch (error) {
    console.error("[router] view failed to load", error);
    return;
  }

  // A slower earlier navigation must not overwrite a faster later one.
  if (token !== renderToken) return;

  rememberOffset();
  const firstPaint = current === null;
  current = path;

  outlet.replaceChildren(view instanceof Node ? view : document.createTextNode(""));
  outlet.classList.remove("view-enter");
  // Force a reflow so the animation restarts on same-class re-entry.
  void outlet.offsetWidth;
  outlet.classList.add("view-enter");

  const restored = history.state?.restore ? offsets.get(path) : undefined;
  window.scrollTo({ top: restored ?? 0, behavior: "auto" });

  // Move focus to the new view for screen-reader and keyboard users, without
  // adding the heading to the tab order permanently.
  //
  // Not on the first paint: there is no *previous* view to have moved away from,
  // and before any pointer input Chrome counts a programmatic focus as
  // keyboard-driven, so the landing page would open with a ring drawn round its
  // own headline.
  const heading = firstPaint ? null : outlet.querySelector("h1, h2, [data-autofocus]");
  if (heading) {
    heading.setAttribute("tabindex", "-1");
    heading.focus({ preventScroll: true });
    heading.addEventListener("blur", () => heading.removeAttribute("tabindex"), { once: true });
  }

  document.dispatchEvent(new CustomEvent("route:changed", { detail: { path, params } }));
}

/** Delegated link handling: any in-app `<a href="/...">` becomes a soft
 *  navigation, so views never need their own click plumbing. */
function onClick(event) {
  if (event.defaultPrevented || event.button !== 0) return;
  if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;

  const link = event.target.closest("a");
  if (!link) return;
  if (link.target === "_blank" || link.hasAttribute("download") || link.dataset.native === "true") return;

  const href = link.getAttribute("href");
  if (!href || href.startsWith("#") || href.startsWith("mailto:") || href.startsWith("tel:")) return;

  const url = new URL(href, location.href);
  if (url.origin !== location.origin) return;

  // /media/* and /docs are served by the backend, not the SPA.
  if (url.pathname.startsWith("/media/") || url.pathname.startsWith("/docs")) return;

  event.preventDefault();
  navigate(url.pathname + url.search);
}

export function start(node) {
  outlet = node;
  // Manual, because we restore per-path offsets ourselves above.
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  document.addEventListener("click", onClick);
  window.addEventListener("popstate", () => {
    history.replaceState({ ...(history.state || {}), restore: true }, "");
    render();
  });
  return render();
}
