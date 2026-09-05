/* Tiny DOM layer. Not a framework — just the three things hand-written views
 * need constantly, with the XSS-safe path as the *default* one.
 *
 * Every string that reaches the DOM here goes through `textContent` or
 * `setAttribute`, never `innerHTML`. Meal data contains user-supplied notes and
 * label corrections, so an `innerHTML` template would be an injection sink.
 * `html()` exists for static markup only and is never given interpolated data.
 */

/** Create an element. Children may be nodes, strings, or nested arrays. */
export function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);

  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;

    if (key === "class") {
      node.className = Array.isArray(value) ? value.filter(Boolean).join(" ") : String(value);
    } else if (key === "style" && typeof value === "object") {
      // setProperty rather than Object.assign: custom properties (`--x`) are
      // not real CSSStyleDeclaration keys and Object.assign drops them.
      for (const [prop, val] of Object.entries(value)) {
        if (val === null || val === undefined) continue;
        if (prop.startsWith("--")) node.style.setProperty(prop, String(val));
        else node.style[prop] = val;
      }
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "html") {
      // Explicit, greppable opt-in. Static markup only.
      node.innerHTML = value;
    } else if (key === "text") {
      node.textContent = String(value);
    } else if (value === true) {
      node.setAttribute(key, "");
    } else {
      node.setAttribute(key, String(value));
    }
  }

  append(node, children);
  return node;
}

/** Namespaced element creator — SVG needs createElementNS or nothing renders. */
export function svg(tag, props = {}, ...children) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on") && typeof value === "function") {
      node.addEventListener(key.slice(2).toLowerCase(), value);
    } else if (key === "class") {
      node.setAttribute("class", Array.isArray(value) ? value.filter(Boolean).join(" ") : value);
    } else if (key === "dataset") {
      Object.assign(node.dataset, value);
    } else if (key === "text") {
      // SVG has no attribute named `text`. Without this branch a <text> node
      // silently renders empty, which is how the donut lost its centre label.
      node.textContent = String(value);
    } else {
      node.setAttribute(key, String(value));
    }
  }
  append(node, children);
  return node;
}

function append(parent, children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false || child === true) continue;
    parent.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
}

/** Replace all children in one shot. */
export function mount(parent, ...children) {
  parent.replaceChildren();
  append(parent, children);
  return parent;
}

export const $ = (selector, root = document) => root.querySelector(selector);
export const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

/** Parse a static markup string into a fragment. Never call with user data. */
export function html(markup) {
  return document.createRange().createContextualFragment(markup);
}

/* -------------------------------------------------------------------- Icons */

/* One 24×24 stroke set, drawn from a single path dictionary so weight and
 * corner radius stay consistent. Inline rather than a sprite sheet: there are
 * ~30 of them and it saves a request plus a fetch-ordering problem on first
 * paint. */
const PATHS = {
  camera:
    "M14.5 4h-5L8 6.5H5A2 2 0 0 0 3 8.5v9a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-9a2 2 0 0 0-2-2h-3L14.5 4Z M12 16.5a3.5 3.5 0 1 0 0-7 3.5 3.5 0 0 0 0 7Z",
  upload: "M12 16V4m0 0L7.5 8.5M12 4l4.5 4.5M4 16v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2",
  image:
    "M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1Z M3 16l4.5-4.5a2 2 0 0 1 2.8 0L15 16 M14 12l1.6-1.6a2 2 0 0 1 2.8 0L21 13 M9 9.5a1 1 0 1 1-2 0 1 1 0 0 1 2 0Z",
  home: "M4 10.5 12 4l8 6.5V19a1 1 0 0 1-1 1h-4v-6H9v6H5a1 1 0 0 1-1-1v-8.5Z",
  history: "M3.5 12a8.5 8.5 0 1 0 2.6-6.1M3.5 4.5V10h5.5M12 8v4.4l3 1.8",
  chart: "M4 20V10m5 10V4m5 16v-7m5 7V8",
  user: "M12 11.5a3.75 3.75 0 1 0 0-7.5 3.75 3.75 0 0 0 0 7.5ZM4.5 20.5a7.5 7.5 0 0 1 15 0",
  settings:
    "M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z M19.4 14.5a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-2.87 1.2v.17a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-2.93-1.18l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.7 1.7 0 0 0 3.5 13.6H3.3a2 2 0 1 1 0-4h.09A1.7 1.7 0 0 0 4.6 6.67l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.87.34h.08A1.7 1.7 0 0 0 10.4 2.7V2.5a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 2.87 1.18l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87v.08a1.7 1.7 0 0 0 1.55 1.02h.17a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.55 1.02Z",
  check: "M4.5 12.5 9 17l10.5-10.5",
  x: "M6 6l12 12M18 6L6 18",
  chevronRight: "M9 5l7 7-7 7",
  chevronLeft: "M15 5l-7 7 7 7",
  chevronDown: "M6 9l6 6 6-6",
  alert: "M12 8.5v5m0 3.2v.3M10.3 3.9 2.6 17.2A1.7 1.7 0 0 0 4.1 19.8h15.8a1.7 1.7 0 0 0 1.5-2.6L13.7 3.9a1.7 1.7 0 0 0-3 0Z",
  info: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 11v5.5M12 7.8v.2",
  trash: "M4 7h16M9 7V4.8A.8.8 0 0 1 9.8 4h4.4a.8.8 0 0 1 .8.8V7M6 7l.9 12.2a1 1 0 0 0 1 .8h8.2a1 1 0 0 0 1-.8L18 7M10 11v5M14 11v5",
  edit: "M4 20h4L19.3 8.7a2 2 0 0 0 0-2.8l-1.2-1.2a2 2 0 0 0-2.8 0L4 16v4Z",
  sparkles:
    "M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3Z M18.5 15l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8.8-2.2Z",
  flame: "M12 21c3.6 0 6.3-2.4 6.3-5.9 0-4.6-4.4-6-4.4-9.8 0-1.2.4-2.3.4-2.3S9 4.6 9 8.4c0 1.3.7 2.2.7 2.2S8 9.4 8 7.3c-1.4 1.6-2.3 3.6-2.3 5.9C5.7 18.2 8.4 21 12 21Z",
  target: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18Zm0-4.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9Zm0-3a1.5 1.5 0 1 0 0-3 1.5 1.5 0 0 0 0 3Z",
  scale: "M12 4v16M7 8h10M5.5 8 3 15h5L5.5 8Zm13 0L16 15h5l-2.5-7Z",
  ruler: "M15.5 3.5 20.5 8.5 8.5 20.5 3.5 15.5 15.5 3.5Z M11 8l2 2M8.5 10.5l2 2M6 13l2 2",
  logout: "M15 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h7a2 2 0 0 0 2-2v-2M10 12h11m0 0-3.5-3.5M21 12l-3.5 3.5",
  login: "M9 8V6a2 2 0 0 1 2-2h7a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-7a2 2 0 0 1-2-2v-2M14 12H3m0 0 3.5-3.5M3 12l3.5 3.5",
  moon: "M20 14.4A8.5 8.5 0 1 1 9.6 4 6.8 6.8 0 0 0 20 14.4Z",
  sun: "M12 16.5a4.5 4.5 0 1 0 0-9 4.5 4.5 0 0 0 0 9ZM12 2v2.2M12 19.8V22M4.2 4.2l1.6 1.6M18.2 18.2l1.6 1.6M2 12h2.2M19.8 12H22M4.2 19.8l1.6-1.6M18.2 5.8l1.6-1.6",
  book: "M4 5.5A1.5 1.5 0 0 1 5.5 4H10a2 2 0 0 1 2 2v14a2 2 0 0 0-2-2H5.5A1.5 1.5 0 0 1 4 16.5v-11Z M20 5.5A1.5 1.5 0 0 0 18.5 4H14a2 2 0 0 0-2 2v14a2 2 0 0 1 2-2h4.5a1.5 1.5 0 0 0 1.5-1.5v-11Z",
  plus: "M12 5v14M5 12h14",
  minus: "M5 12h14",
  refresh: "M20 11a8 8 0 1 0-2.5 6M20 5.5V11h-5.5",
  eye: "M12 5c5 0 9 7 9 7s-4 7-9 7-9-7-9-7 4-7 9-7Zm0 9.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z",
  eyeOff: "M4 4l16 16M9.9 5.4A9 9 0 0 1 12 5c5 0 9 7 9 7a17 17 0 0 1-2.2 2.9M6.4 7.6A17 17 0 0 0 3 12s4 7 9 7a9 9 0 0 0 3-.5M10.2 10.3a2.5 2.5 0 0 0 3.5 3.5",
  cube: "M12 3l8 4.5v9L12 21l-8-4.5v-9L12 3Zm0 0v18M4 7.5l8 4.5 8-4.5",
  layers: "M12 3 3 8l9 5 9-5-9-5ZM3 13l9 5 9-5M3 17.5l9 5 9-5",
  brain: "M9.5 4A3.5 3.5 0 0 0 6 7.5v.6A3 3 0 0 0 4 11c0 1.1.6 2 1.5 2.6A3 3 0 0 0 8 18h1.5V4Zm5 0A3.5 3.5 0 0 1 18 7.5v.6A3 3 0 0 1 20 11c0 1.1-.6 2-1.5 2.6A3 3 0 0 1 16 18h-1.5V4Z",
  clock: "M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18ZM12 7.5V12l3 2",
  arrowLeft: "M19 12H5m0 0 6-6M5 12l6 6",
};

/** Inline SVG icon. `name` must be a key of PATHS; unknown names render nothing. */
export function icon(name, { size, filled = false, ...rest } = {}) {
  const path = PATHS[name];
  if (!path) return svg("svg", { viewBox: "0 0 24 24", "aria-hidden": "true" });
  return svg(
    "svg",
    {
      viewBox: "0 0 24 24",
      fill: filled ? "currentColor" : "none",
      stroke: filled ? "none" : "currentColor",
      "stroke-width": 1.8,
      "stroke-linecap": "round",
      "stroke-linejoin": "round",
      "aria-hidden": "true",
      ...(size ? { width: size, height: size } : {}),
      ...rest,
    },
    svg("path", { d: path })
  );
}

export const ICON_NAMES = Object.keys(PATHS);

/* ------------------------------------------------------------- Formatting */

/* Locale-aware but deliberately not locale-*dependent* for units: grams and
 * kcal are the same words in every locale this app targets, and hard-coding
 * them keeps the numbers aligned in the tabular columns. */

const nf = (digits) =>
  new Intl.NumberFormat(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });

const NF0 = nf(0);
const NF1 = nf(1);

export function num(value, digits = 0) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "—";
  return (digits === 0 ? NF0 : digits === 1 ? NF1 : nf(digits)).format(n);
}

/** Calories: never fractional. A "412.7 kcal" reading implies precision we
 *  do not have — the underlying weight estimate is ±25%. */
export const kcal = (value) => num(Math.round(Number(value) || 0), 0);

/** Grams: one decimal below 10 g, whole numbers above. Sub-gram precision
 *  matters for spices and condiments, and is noise for a bowl of rice. */
export function grams(value) {
  const n = Number(value) || 0;
  return n > 0 && n < 10 ? num(n, 1) : num(Math.round(n), 0);
}

/** Micronutrients arrive in mg or µg; the magnitude spans 6 orders. */
export function micro(value) {
  const n = Number(value) || 0;
  if (n === 0) return "0";
  if (n < 0.1) return num(n, 2);
  if (n < 10) return num(n, 1);
  return num(Math.round(n), 0);
}

export const pct = (value) => `${num(Math.round(Number(value) || 0), 0)}%`;

const UNIT_LABEL = { g: "g", mg: "mg", mcg: "µg" };

/** Split a nutrient key like `vitamin_b12_mcg` into a label and its unit. */
export function nutrientLabel(key) {
  const parts = String(key).split("_");
  const last = parts[parts.length - 1];
  const unit = UNIT_LABEL[last] ? parts.pop() : "";
  const name = parts
    .map((word) => (word.length <= 2 ? word.toUpperCase() : word[0].toUpperCase() + word.slice(1)))
    .join(" ")
    .replace(/^Vitamin B12$/i, "Vitamin B12");
  return { name, unit: UNIT_LABEL[unit] || "" };
}

/** "just now" / "14:32" / "Yesterday 09:10" / "12 Aug" — a history list reads
 *  as a timeline, so recency deserves more resolution than absolute dates. */
export function when(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const diffMs = now - date;
  const time = date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });

  if (diffMs < 60_000) return "just now";
  if (diffMs < 3_600_000) return `${Math.floor(diffMs / 60_000)} min ago`;

  const sameDay = date.toDateString() === now.toDateString();
  if (sameDay) return time;

  const yesterday = new Date(now);
  yesterday.setDate(now.getDate() - 1);
  if (date.toDateString() === yesterday.toDateString()) return `Yesterday ${time}`;

  const sameYear = date.getFullYear() === now.getFullYear();
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    ...(sameYear ? {} : { year: "numeric" }),
  });
}

export function dayLabel(isoDate) {
  const date = new Date(`${isoDate}T12:00:00`);
  return Number.isNaN(date.getTime()) ? "" : date.toLocaleDateString(undefined, { weekday: "narrow" });
}

export function dayFull(isoDate) {
  const date = new Date(`${isoDate}T12:00:00`);
  return Number.isNaN(date.getTime())
    ? isoDate
    : date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
}

/** Turn a snake_case label into something displayable, for the rare case where
 *  the backend has no display name for it. */
export function humanize(label) {
  return String(label || "")
    .split(/[_\s]+/)
    .filter(Boolean)
    .map((word) => word[0].toUpperCase() + word.slice(1))
    .join(" ");
}

export const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

/** Debounce, for the settings inputs that PATCH as you type. */
export function debounce(fn, ms = 400) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}

export const prefersReducedMotion = () =>
  window.matchMedia("(prefers-reduced-motion: reduce)").matches;
