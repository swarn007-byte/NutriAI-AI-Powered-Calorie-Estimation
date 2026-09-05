/* History — every meal analysed, newest first.
 *
 * The backend paginates but does not filter by date, so the `?date=` links that
 * the Today trend bars produce are resolved here: one wide page is fetched and
 * narrowed client-side. That is a deliberate trade — a day filter that only
 * searched the current 30-meal page would silently lie about older days.
 */

import { el, icon, kcal, grams, num, when, dayFull, mount } from "../dom.js";
import { navigate } from "../router.js";
import { api } from "../api.js";
import { state, set, isGuest } from "../store.js";
import { toastError } from "../toast.js";

const PAGE = 24;
/** Wide enough to cover the 14-day trend the dashboard links into. */
const FILTER_WINDOW = 100;
/** `limit` is capped at 100 by the endpoint (main.py:556). */
const SERVER_MAX = 100;

export default function history({ query } = {}) {
  const date = query?.date || null;
  const page = el("div", { class: "page stack stack--lg" });
  const body = el("div", { class: "stack stack--lg" });

  page.append(header(date), body);

  let offset = 0;
  let loaded = [];
  let total = 0;
  let busy = false;

  function paint() {
    const rows = date ? loaded.filter((meal) => localDate(meal.captured_at) === date) : loaded;

    if (!rows.length) {
      body.replaceChildren(busy ? skeleton() : empty(date, total));
      return;
    }

    // mount(), not replaceChildren(): the pager is absent when a single day is
    // filtered, and a null child would render as the text "null".
    mount(
      body,
      summaryLine(rows, date, total),
      el("div", { class: "history" }, rows.map(mealCard)),
      date ? null : pager({ shown: loaded.length, total, busy, onMore: more })
    );
  }

  async function load({ append = false } = {}) {
    busy = true;
    paint();
    // On a refresh (not a "load more") re-request everything already on screen,
    // or the list would visibly shrink back to one page.
    const limit = append
      ? PAGE
      : Math.min(SERVER_MAX, Math.max(date ? FILTER_WINDOW : PAGE, loaded.length));
    try {
      const payload = await api.history({ limit, offset: append ? offset : 0 });
      if (!page.isConnected) return;
      total = payload.total || 0;
      loaded = append ? [...loaded, ...(payload.meals || [])] : payload.meals || [];
      set({ history: { total, meals: loaded } });
    } catch (error) {
      if (!page.isConnected) return;
      if (!loaded.length) {
        busy = false;
        body.replaceChildren(failure(error));
        return;
      }
      toastError(append ? "Couldn't load more meals." : "Couldn't refresh your history.");
    } finally {
      busy = false;
    }
    paint();
  }

  function more() {
    if (busy) return;
    offset = loaded.length;
    load({ append: true });
  }

  const cached = state.history;
  if (cached && !date) {
    loaded = cached.meals;
    total = cached.total;
  }
  load();

  return page;
}

/** The API returns UTC timestamps; the day a meal *belongs* to is the local one,
 *  which is also how the summary endpoint buckets its trend. */
function localDate(iso) {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

/* ------------------------------------------------------------------- Header */

function header(date) {
  return el(
    "div",
    { class: "row row--between" },
    el(
      "div",
      {},
      el("span", { class: "eyebrow", text: date ? "Filtered" : "Every meal you've logged" }),
      el("h1", { style: { fontSize: "var(--step-3)" }, text: date ? dayFull(date) : "History" })
    ),
    el(
      "div",
      { class: "row row--tight" },
      date
        ? el(
            "a",
            { href: "/history", class: "btn btn--outline btn--sm" },
            icon("x", { size: 15 }),
            el("span", { text: "Clear filter" })
          )
        : null,
      el(
        "a",
        { href: "/", class: "btn btn--primary btn--sm" },
        icon("plus", { size: 16 }),
        el("span", { text: "Log a meal" })
      )
    )
  );
}

function summaryLine(rows, date, total) {
  const calories = rows.reduce((acc, meal) => acc + (Number(meal.total_calories) || 0), 0);
  const protein = rows.reduce((acc, meal) => acc + (Number(meal.total_protein_g) || 0), 0);

  return el(
    "div",
    { class: "row row--between small" },
    el("span", {
      class: "muted",
      text: date
        ? `${rows.length} meal${rows.length === 1 ? "" : "s"} on this day`
        : `Showing ${rows.length} of ${num(total)} meal${total === 1 ? "" : "s"}`,
    }),
    el("span", { class: "faint nowrap", text: `${kcal(calories)} kcal · ${grams(protein)} g protein` })
  );
}

/* ---------------------------------------------------------------- Meal card */

function mealCard(meal) {
  return el(
    "a",
    { href: `/meal/${meal.meal_id}`, class: "card card--interactive mealcard" },
    el(
      "div",
      { class: "mealcard__media" },
      meal.thumb_url
        ? el("img", {
            src: meal.thumb_url,
            alt: meal.top_items?.length ? meal.top_items.join(", ") : "Analysed meal",
            loading: "lazy",
            decoding: "async",
          })
        : el("div", { class: "center full" }, icon("image", { size: 24 })),
      el("span", { class: "mealcard__kcal", text: `${kcal(meal.total_calories)} kcal` }),
      meal.has_low_confidence
        ? el(
            "span",
            { class: "mealcard__flag" },
            el("span", { class: "chip chip--warn", text: "Unsure" })
          )
        : null
    ),
    el(
      "div",
      { class: "mealcard__body" },
      el("span", { class: "mealcard__when", text: when(meal.captured_at) }),
      el("div", {
        class: "mealcard__items clamp-2",
        text: meal.top_items?.length ? meal.top_items.join(" · ") : "No items identified",
      }),
      el(
        "div",
        { class: "mealcard__foot row row--between xsmall faint" },
        el("span", {
          text: `P ${grams(meal.total_protein_g)} · C ${grams(meal.total_carbs_g)} · F ${grams(meal.total_fat_g)}`,
        }),
        el("span", { text: `${meal.item_count} item${meal.item_count === 1 ? "" : "s"}` })
      )
    )
  );
}

/* -------------------------------------------------------------- Pagination */

/** A "load more" button rather than infinite scroll: this list is something
 *  people scan for a specific meal, and a footer that keeps running away is
 *  hostile to that. */
function pager({ shown, total, busy, onMore }) {
  if (shown >= total) {
    return shown > PAGE
      ? el("p", { class: "center small faint", text: "That's everything." })
      : null;
  }

  const button = el(
    "button",
    { class: "btn btn--outline btn--block", disabled: busy || null },
    icon("chevronDown", { size: 16 }),
    el("span", { text: busy ? "Loading…" : `Load ${Math.min(PAGE, total - shown)} more` })
  );
  button.addEventListener("click", onMore);
  return button;
}

/* ------------------------------------------------------------ Empty / error */

function skeleton() {
  return el(
    "div",
    { class: "history", "aria-busy": "true" },
    Array.from({ length: 6 }, () => el("div", { class: "skeleton", style: { height: "14rem" } }))
  );
}

function empty(date, total) {
  if (date) {
    return el(
      "div",
      { class: "empty" },
      el("div", { class: "empty__icon" }, icon("history", { size: 26 })),
      el("h2", { class: "empty__title", text: "Nothing logged that day" }),
      el("p", {
        class: "empty__text",
        text: `No meals were analysed on ${dayFull(date)}${total ? ", though there are meals on other days." : "."}`,
      }),
      el(
        "div",
        { class: "row", style: { justifyContent: "center" } },
        el("a", { href: "/history", class: "btn btn--primary" }, el("span", { text: "See all meals" }))
      )
    );
  }

  return el(
    "div",
    { class: "stack stack--lg" },
    el(
      "div",
      { class: "empty" },
      el("div", { class: "empty__icon" }, icon("history", { size: 26 })),
      el("h2", { class: "empty__title", text: "No meals yet" }),
      el("p", {
        class: "empty__text",
        text: "Analysed meals collect here with their photo, calories and macros. Nothing is uploaded until you choose a photo.",
      }),
      el(
        "div",
        { class: "row", style: { justifyContent: "center" } },
        el("a", { href: "/", class: "btn btn--primary" }, icon("camera", { size: 16 }), el("span", { text: "Analyse a meal" })),
        el("a", { href: "/method", class: "btn btn--ghost" }, icon("book", { size: 16 }), el("span", { text: "How it works" }))
      )
    ),
    isGuest()
      ? el(
          "div",
          { class: "panel panel--info" },
          el(
            "div",
            { class: "row row--between" },
            el(
              "div",
              { class: "row row--tight" },
              icon("info", { size: 16 }),
              el("strong", { text: "You're browsing as a guest" })
            ),
            el("a", { href: "/auth", class: "btn btn--sm btn--outline", text: "Create an account" })
          ),
          el("p", {
            class: "small muted",
            style: { marginTop: "0.35rem" },
            text: "Guest meals live in this browser's session token. Creating an account keeps everything you've already analysed and makes it reachable from another device.",
          })
        )
      : null
  );
}

function failure(error) {
  const retry = el(
    "button",
    { class: "btn btn--primary" },
    icon("refresh", { size: 16 }),
    el("span", { text: "Try again" })
  );
  retry.addEventListener("click", () => navigate("/history", { replace: true }));

  return el(
    "div",
    { class: "empty" },
    el("div", { class: "empty__icon" }, icon("alert", { size: 26 })),
    el("h2", { class: "empty__title", text: "Couldn't load your history" }),
    el("p", { class: "empty__text", text: error?.message || "" }),
    el("div", { class: "row", style: { justifyContent: "center" } }, retry)
  );
}
