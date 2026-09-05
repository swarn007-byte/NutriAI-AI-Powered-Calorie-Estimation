/* 404 — a mistyped or dead URL.
 *
 * Reachable in one way people don't expect: the backend serves index.html for
 * every unmatched path (main.py:689), so a hand-typed /histry loads the whole
 * app and lands here rather than showing a server error page. That means this
 * view has to be a real part of the app — navigation intact, a way onward — not
 * a dead end.
 */

import { el, icon } from "../dom.js";
import { state } from "../store.js";

/** Cheap edit distance, capped: only used to suggest one of nine known routes,
 *  so the quadratic cost is over strings of a dozen characters. */
function distance(a, b) {
  const rows = Array.from({ length: b.length + 1 }, (_, i) => i);
  for (let i = 1; i <= a.length; i += 1) {
    let previous = rows[0];
    rows[0] = i;
    for (let j = 1; j <= b.length; j += 1) {
      const carry = rows[j];
      rows[j] = Math.min(
        rows[j] + 1,
        rows[j - 1] + 1,
        previous + (a[i - 1] === b[j - 1] ? 0 : 1)
      );
      previous = carry;
    }
  }
  return rows[b.length];
}

const ROUTES = [
  { path: "/", label: "Analyse a meal", iconName: "camera" },
  { path: "/today", label: "Today", iconName: "target" },
  { path: "/history", label: "History", iconName: "history" },
  { path: "/method", label: "How it works", iconName: "book" },
  { path: "/settings", label: "Settings", iconName: "settings" },
];

/** The closest real route, if the typo is small enough to be worth guessing at.
 *  Three edits on a short path is already a different word, not a slip. */
function suggest(path) {
  const target = path.replace(/\/+$/, "").toLowerCase() || "/";
  let best = null;
  for (const route of ROUTES) {
    if (route.path === "/") continue;
    const d = distance(target, route.path);
    if (d > 0 && d <= 3 && (!best || d < best.d)) best = { ...route, d };
  }
  return best;
}

export default function notfound({ path } = {}) {
  const attempted = typeof path === "string" ? path : "/";
  const near = suggest(attempted);
  const secondary = state.user ? "/today" : "/method";

  /* The chip row is "everywhere else". A destination already offered as a button
   * a few inches above does not need repeating, and the repetition made the two
   * rows look like the same list rendered twice. */
  const offered = new Set(["/", secondary, near?.path].filter(Boolean));

  return el(
    "div",
    { class: "page" },
    el(
      "div",
      { class: "notfound" },
      el("span", { class: "notfound__code", text: "404" }),
      el("h1", { style: { fontSize: "var(--step-3)" }, text: "That page doesn't exist" }),

      /* The path is shown so a mistyped or stale link is diagnosable, and it is
       * set as text — a crafted URL is untrusted input and never becomes markup. */
      el(
        "p",
        { class: "muted small", style: { maxWidth: "44ch" } },
        el("span", { text: "Nothing is routed at " }),
        el("code", { text: attempted }),
        el("span", { text: ". It may have been a typo, or a link from an older version of the app." })
      ),

      near
        ? el(
            "div",
            { class: "row row--tight", style: { justifyContent: "center" } },
            el("span", { class: "small faint", text: "Did you mean" }),
            el("a", { href: near.path, class: "btn btn--outline btn--sm" }, el("span", { text: near.label })),
            el("span", { class: "small faint", text: "?" })
          )
        : null,

      el(
        "div",
        { class: "row", style: { justifyContent: "center", marginTop: "var(--space-2xs)" } },
        el(
          "a",
          { href: "/", class: "btn btn--primary" },
          icon("camera", { size: 16 }),
          el("span", { text: "Analyse a meal" })
        ),
        el(
          "a",
          { href: secondary, class: "btn btn--outline" },
          icon(state.user ? "target" : "book", { size: 16 }),
          el("span", { text: state.user ? "Go to Today" : "How it works" })
        )
      ),

      el("div", { class: "divider", style: { maxWidth: "22rem", width: "100%" } }),

      el(
        "div",
        { class: "row row--tight", style: { justifyContent: "center" } },
        ROUTES.filter((route) => !offered.has(route.path)).map((route) =>
          el(
            "a",
            { href: route.path, class: "chip", style: { textDecoration: "none" } },
            icon(route.iconName, { size: 13 }),
            el("span", { text: route.label })
          )
        )
      )
    )
  );
}
