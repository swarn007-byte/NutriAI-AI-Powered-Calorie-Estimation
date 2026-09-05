/* App shell: sidebar, top bar, bottom tab bar.
 *
 * One nav definition drives all three, so a destination can never appear in the
 * sidebar and go missing from the tab bar.
 */

import { el, svg, icon } from "../dom.js";
import { navigate } from "../router.js";
import { state, subscribe, setTheme, displayName, isGuest } from "../store.js";

const NAV = [
  { path: "/", label: "Analyse", icon: "camera", tab: true },
  { path: "/today", label: "Today", icon: "target", tab: true },
  { path: "/history", label: "History", icon: "history", tab: true },
  { path: "/method", label: "Method", icon: "book", tab: false },
  { path: "/settings", label: "Settings", icon: "settings", tab: true },
];

const isActive = (path, current) =>
  path === "/" ? current === "/" : current === path || current.startsWith(`${path}/`);

function navLink(entry, current, className) {
  return el(
    "a",
    {
      href: entry.path,
      class: className,
      "aria-current": isActive(entry.path, current) ? "page" : null,
    },
    icon(entry.icon),
    el("span", { text: entry.label })
  );
}

export function sidebar() {
  const nav = el("nav", { class: "sidebar__nav", "aria-label": "Main" });
  const foot = el("div", { class: "sidebar__foot" });

  const node = el(
    "aside",
    { class: "sidebar" },
    el(
      "a",
      { href: "/", class: "sidebar__brand", "aria-label": "Nutri-AI home" },
      logo(28),
      el("span", { text: "Nutri-AI" })
    ),
    nav,
    foot
  );

  const paint = () => {
    const current = location.pathname;
    nav.replaceChildren(...NAV.map((entry) => navLink(entry, current, "navlink")));
    foot.replaceChildren(themeToggle(), accountLink());
  };

  paint();
  document.addEventListener("route:changed", paint);
  subscribe((keys) => {
    if (keys.includes("user") || keys.includes("theme")) paint();
  });
  return node;
}

export function tabbar() {
  const node = el("nav", { class: "tabbar", "aria-label": "Main" });

  const paint = () => {
    const current = location.pathname;
    node.replaceChildren(
      ...NAV.filter((entry) => entry.tab).map((entry) => navLink(entry, current, "tab"))
    );
  };

  paint();
  document.addEventListener("route:changed", paint);
  return node;
}

export function topbar() {
  const title = el("div", { class: "topbar__title strong hide-sm" });
  const actions = el("div", { class: "topbar__actions" });

  const node = el(
    "header",
    { class: "topbar" },
    el(
      "a",
      { href: "/", class: "topbar__brand", "aria-label": "Nutri-AI home" },
      logo(26),
      el("span", { text: "Nutri-AI" })
    ),
    title,
    el("div", { class: "topbar__spacer" }),
    actions
  );

  const paint = () => {
    const entry = NAV.find((row) => isActive(row.path, location.pathname));
    title.textContent = entry && entry.path !== "/" ? entry.label : "";

    const children = [engineChip()];
    // The theme toggle lives in the sidebar on desktop; on mobile there is no
    // sidebar, so it moves up here rather than being buried in Settings.
    if (!window.matchMedia("(min-width: 1024px)").matches) children.push(themeToggle(true));
    children.push(
      isGuest()
        ? el(
            "a",
            { href: "/auth", class: "btn btn--outline btn--sm", "aria-label": "Sign in" },
            icon("login", { size: 15 }),
            el("span", { class: "hide-sm", text: "Sign in" })
          )
        : el(
            "a",
            {
              href: "/settings",
              class: "btn btn--ghost btn--icon",
              "aria-label": `Account: ${displayName()}`,
              "data-tip": displayName(),
            },
            avatar(displayName())
          )
    );
    actions.replaceChildren(...children);
  };

  paint();
  document.addEventListener("route:changed", paint);
  subscribe((keys) => {
    if (keys.includes("user") || keys.includes("theme") || keys.includes("health")) paint();
  });
  return node;
}

/* --------------------------------------------------------------- Fragments */

/** The engine badge is not decoration: it tells the user whether they are
 *  looking at trained-model output or the heuristic fallback, which changes how
 *  much the numbers should be trusted (design.md §20). */
function engineChip() {
  const engine = state.health?.engine;
  if (!engine) return el("span");
  const full = engine === "full";
  return el(
    "a",
    {
      href: "/method",
      class: ["chip", "chip--dot", full ? "chip--brand" : ""],
      // The visible label is `.hide-sm`, so below 640px this collapses to a bare
      // dot and the link is left with no accessible name at all. Name it here
      // rather than leaning on text that a media query takes away.
      "aria-label": full
        ? "Running the trained model stack — read how this works"
        : "Running the built-in estimator, no trained weights installed — read how this works",
      "data-tip": full
        ? "Running the trained model stack"
        : "Running the built-in estimator — no trained weights installed. Tap to read how this works.",
    },
    el("span", { class: "hide-sm", text: full ? "Model stack" : "Estimator" })
  );
}

function themeToggle(iconOnly = false) {
  const light = document.documentElement.dataset.theme === "light";
  return el(
    "button",
    {
      class: iconOnly ? "btn btn--ghost btn--icon" : "navlink",
      "aria-label": `Switch to ${light ? "dark" : "light"} theme`,
      "data-tip": iconOnly ? `${light ? "Dark" : "Light"} theme` : null,
      onclick: () => setTheme(light ? "dark" : "light"),
    },
    icon(light ? "moon" : "sun"),
    iconOnly ? null : el("span", { text: light ? "Dark theme" : "Light theme" })
  );
}

function accountLink() {
  if (isGuest()) {
    return el(
      "a",
      { href: "/auth", class: "navlink" },
      icon("login"),
      el("span", { text: "Sign in" })
    );
  }
  return el(
    "a",
    { href: "/settings", class: "navlink" },
    icon("user"),
    el("span", { class: "truncate", text: displayName() })
  );
}

function avatar(name) {
  const initial = (name || "?").trim().charAt(0).toUpperCase();
  return el("span", {
    text: initial,
    style: {
      display: "grid",
      placeItems: "center",
      width: "28px",
      height: "28px",
      borderRadius: "50%",
      background: "var(--brand-soft)",
      border: "1px solid var(--brand-line)",
      color: "var(--brand)",
      fontWeight: "700",
      fontSize: "0.8rem",
    },
  });
}

/** The mark: a plate ring with a leaf. Inline SVG so it inherits currentColor
 *  and needs no asset request on first paint. Built node by node, so raw markup
 *  assignment stays confined to dom.js — worth the verbosity in an app that
 *  renders dish names straight from the API.
 *
 *  The gradient id is per-instance because the topbar and a page can both hold a
 *  logo at once, and a duplicate id would point the second at the first's fill. */
let logoSeq = 0;

export function logo(size = 28) {
  const gradientId = `logo-grad-${(logoSeq += 1)}`;
  const fill = `url(#${gradientId})`;

  return svg(
    "svg",
    { viewBox: "0 0 32 32", width: size, height: size, "aria-hidden": "true", style: "flex:none" },
    svg(
      "defs",
      {},
      svg(
        "linearGradient",
        { id: gradientId, x1: "0", y1: "0", x2: "1", y2: "1" },
        svg("stop", { offset: "0%", "stop-color": "var(--brand)" }),
        svg("stop", { offset: "100%", "stop-color": "var(--accent)" })
      )
    ),
    svg("circle", { cx: 16, cy: 16, r: 13.4, fill: "none", stroke: fill, "stroke-width": 2.4 }),
    svg("path", {
      d: "M22 9.6c0 6.4-3.9 10.2-8.6 10.2-1.2 0-2.3-.3-2.3-.3s.4-6 4.2-8.2c2.5-1.5 6.7-1.7 6.7-1.7Z",
      fill,
    }),
    svg("path", {
      d: "M10 23.2c0-4.2 2.2-8 5.6-10",
      fill: "none",
      stroke: "var(--brand)",
      "stroke-width": 2.1,
      "stroke-linecap": "round",
    })
  );
}

/** Skip link + live region, mounted once by main.js. */
export function a11yLayer() {
  return [
    el("a", { class: "skip-link", href: "#main", text: "Skip to content" }),
    el("div", {
      id: "live",
      class: "sr-only",
      role: "status",
      "aria-live": "polite",
      "aria-atomic": "true",
    }),
  ];
}

/** Announce a message to screen readers without showing a toast. Used for
 *  progress transitions during analysis, where a visual stepper carries the
 *  information for sighted users. */
export function announce(message) {
  const region = document.getElementById("live");
  if (region) region.textContent = message;
}

export function backLink(to, label) {
  return el(
    "button",
    {
      class: "btn btn--ghost btn--sm",
      onclick: () => (history.length > 1 ? history.back() : navigate(to)),
    },
    icon("arrowLeft", { size: 16 }),
    el("span", { text: label })
  );
}
