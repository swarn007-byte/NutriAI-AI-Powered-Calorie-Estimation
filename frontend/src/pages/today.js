/* Today — the dashboard. Answers one question first ("how am I doing against my
 * goal right now?") and supports it with the trend, the macro split and the
 * micronutrient picture.
 *
 * The gauge is deliberately the only thing above the fold. A dashboard that
 * opens with six equal-weight cards makes the reader do the prioritising.
 */

import { el, icon, kcal, grams, num, pct, dayLabel, dayFull } from "../dom.js";
import { gauge, trendBars, sparkline, MACROS } from "../charts.js";
import { macroList, microGrid } from "../components/meal.js";
import { navigate } from "../router.js";
import { api } from "../api.js";
import { state, set, isGuest, displayName } from "../store.js";
import { toastError } from "../toast.js";

export default function today() {
  const page = el("div", { class: "page stack stack--lg" });
  const body = el("div", { class: "stack stack--lg" });

  page.append(header(), body);

  const cached = state.summary;
  if (cached) {
    body.replaceChildren(...dashboard(cached));
  } else {
    body.replaceChildren(skeleton());
  }

  api
    .summary({ days: 14 })
    .then((summary) => {
      set({ summary });
      if (page.isConnected) body.replaceChildren(...dashboard(summary));
    })
    .catch((error) => {
      if (!page.isConnected) return;
      if (cached) {
        toastError("Couldn't refresh today's numbers.");
        return;
      }
      body.replaceChildren(
        el(
          "div",
          { class: "empty" },
          el("div", { class: "empty__icon" }, icon("alert", { size: 26 })),
          el("h2", { class: "empty__title", text: "Couldn't load your day" }),
          el("p", { class: "empty__text", text: error?.message || "" }),
          el("button", { class: "btn btn--primary", text: "Retry", onclick: () => location.reload() })
        )
      );
    });

  return page;
}

function header() {
  const hour = new Date().getHours();
  const part = hour < 12 ? "Morning" : hour < 17 ? "Afternoon" : "Evening";
  return el(
    "div",
    { class: "row row--between" },
    el(
      "div",
      {},
      el("span", { class: "eyebrow", text: dayFull(new Date().toISOString().slice(0, 10)) }),
      el("h1", { style: { fontSize: "var(--step-3)" }, text: isGuest() ? `Good ${part.toLowerCase()}` : `Good ${part.toLowerCase()}, ${displayName()}` })
    ),
    el(
      "a",
      { href: "/", class: "btn btn--primary btn--sm" },
      icon("plus", { size: 16 }),
      el("span", { text: "Log a meal" })
    )
  );
}

function skeleton() {
  return el(
    "div",
    { class: "stack stack--lg", "aria-busy": "true" },
    el("div", { class: "skeleton", style: { height: "16rem" } }),
    el(
      "div",
      { class: "grid grid--auto" },
      el("div", { class: "skeleton", style: { height: "8rem" } }),
      el("div", { class: "skeleton", style: { height: "8rem" } }),
      el("div", { class: "skeleton", style: { height: "8rem" } })
    ),
    el("div", { class: "skeleton", style: { height: "14rem" } })
  );
}

/* --------------------------------------------------------------- Dashboard */

function dashboard(summary) {
  const goal = summary.goal || {};
  const day = summary.today || {};
  const trend = summary.trend || [];
  const logged = day.meals || 0;

  if (!summary.meals_logged) return [firstRun(summary)];

  return [
    goalCard(day, goal, summary),
    statStrip(summary, trend),
    trendCard(trend, goal),
    macroCard(day, goal),
    microCard(summary),
    logged === 0 ? nudge() : null,
  ].filter(Boolean);
}

/** The headline card. Remaining calories is the number people actually act on,
 *  so it is stated in words next to the gauge rather than left to be subtracted. */
function goalCard(day, goal, summary) {
  const target = Number(goal.calories) || 0;
  const eaten = Number(day.calories) || 0;
  const left = target - eaten;

  return el(
    "section",
    { class: "card card--pad-lg" },
    el(
      "div",
      { class: "split" },
      el(
        "div",
        { class: "center", style: { display: "grid", placeItems: "center" } },
        gauge(eaten, target, { size: 196 })
      ),
      el(
        "div",
        { class: "stack stack--sm" },
        el(
          "div",
          {},
          el("span", { class: "eyebrow", text: left >= 0 ? "Still available" : "Over goal by" }),
          el(
            "div",
            { class: "stat__value", style: { fontSize: "var(--step-4)", color: left >= 0 ? "var(--text)" : "var(--warn)" } },
            el("span", { text: kcal(Math.abs(left)) }),
            el("span", { class: "stat__unit", text: " kcal" })
          ),
          el("p", {
            class: "small muted",
            text:
              target === 0
                ? "Set a calorie goal in Settings to track against it."
                : left >= 0
                  ? `${kcal(eaten)} of ${kcal(target)} logged across ${day.meals || 0} meal${day.meals === 1 ? "" : "s"}.`
                  : `${kcal(eaten)} logged against a ${kcal(target)} goal.`,
          })
        ),
        el("div", { class: "divider" }),
        macroList(day, { goals: goalMacros(goal) }),
        summary.streak_days > 0
          ? el(
              "div",
              { class: "row row--tight" },
              el("span", { class: "chip chip--brand" }, icon("flame", { size: 13 }), el("span", { text: `${summary.streak_days}-day streak` })),
              el("span", { class: "xsmall faint", text: "Consecutive days with a logged meal" })
            )
          : null
      )
    )
  );
}

const goalMacros = (goal) => ({
  protein_g: Number(goal.protein_g) || 0,
  carbs_g: Number(goal.carbs_g) || 0,
  fat_g: Number(goal.fat_g) || 0,
});

/* ------------------------------------------------------------- Stat strip */

function statStrip(summary, trend) {
  const withMeals = trend.filter((row) => row.meals > 0);
  const average = withMeals.length
    ? withMeals.reduce((acc, row) => acc + (row.calories || 0), 0) / withMeals.length
    : 0;
  const best = summary.best_day;

  return el(
    "div",
    { class: "grid grid--auto-sm" },
    stat("Meals logged", num(summary.meals_logged), null, "history"),
    stat("Daily average", kcal(average), "kcal", "chart", withMeals.length ? `Across ${withMeals.length} day${withMeals.length === 1 ? "" : "s"} with a meal` : null),
    stat("Streak", num(summary.streak_days), summary.streak_days === 1 ? "day" : "days", "flame"),
    best
      ? stat("Highest day", kcal(best.calories), "kcal", "target", dayFull(best.date))
      : null
  );
}

function stat(label, value, unit, iconName, note) {
  return el(
    "div",
    { class: "card stat" },
    el(
      "div",
      { class: "row row--between" },
      el("span", { class: "stat__label", text: label }),
      icon(iconName, { size: 15 })
    ),
    el(
      "div",
      { class: "stat__value" },
      el("span", { text: value }),
      unit ? el("span", { class: "stat__unit", text: ` ${unit}` }) : null
    ),
    note ? el("span", { class: "xsmall faint", text: note }) : null
  );
}

/* ------------------------------------------------------------- Trend card */

function trendCard(trend, goal) {
  const days = trend.map((row) => ({
    ...row,
    short: dayLabel(row.date),
    label: dayFull(row.date),
  }));

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Last 14 days" }),
      el("span", { class: "card__note", text: "Tap a day for its meals" })
    ),
    trendBars(days, {
      goal: Number(goal.calories) || 0,
      onSelect: (day) => {
        if (day.meals > 0) navigate(`/history?date=${day.date}`);
      },
    }),
    el("div", { class: "divider" }),
    el(
      "div",
      { class: "row row--between" },
      el("span", { class: "small faint", text: "Protein trend" }),
      el("span", { class: "small faint", text: `${grams(days[days.length - 1]?.protein_g || 0)} g today` })
    ),
    sparkline(days.map((row) => row.protein_g || 0), { color: "var(--protein)" })
  );
}

/* ------------------------------------------------------------- Macro card */

/** Macro *balance*, which is a different question from macro totals: whether
 *  the day's split is in a sensible range, not how many grams it contains. */
function macroCard(day, goal) {
  const total = MACROS.reduce((acc, macro) => acc + (Number(day[macro.key]) || 0) * macro.kcalPerGram, 0);

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Macro balance" }),
      el("span", { class: "card__note", text: total > 0 ? "Share of today's energy" : "Nothing logged yet" })
    ),
    total > 0
      ? el(
          "div",
          { class: "stack stack--sm" },
          el(
            "div",
            { class: "macrobar", style: { height: "10px" } },
            MACROS.map((macro) =>
              el("div", {
                class: "macrobar__seg",
                style: {
                  flexGrow: String((Number(day[macro.key]) || 0) * macro.kcalPerGram),
                  background: macro.color,
                },
              })
            )
          ),
          el(
            "div",
            { class: "row" },
            MACROS.map((macro) => {
              const energy = (Number(day[macro.key]) || 0) * macro.kcalPerGram;
              const target = Number(goal[macro.key]) || 0;
              const actual = Number(day[macro.key]) || 0;
              return el(
                "div",
                { class: "row row--tight grow", style: { minWidth: "8rem" } },
                el("span", { class: "dot", style: { "--c": macro.color } }),
                el(
                  "div",
                  {},
                  el("div", { class: "small strong", text: `${pct((energy / total) * 100)} ${macro.label}` }),
                  el("div", {
                    class: "xsmall faint",
                    text: target ? `${grams(actual)} / ${grams(target)} g` : `${grams(actual)} g`,
                  })
                )
              );
            })
          )
        )
      : el("p", { class: "muted small", text: "Log a meal and the split appears here." })
  );
}

/* ------------------------------------------------------------- Micro card */

function microCard(summary) {
  const values = summary.daily_values || {};
  const short = Object.entries(values)
    .filter(([key, value]) => !CAUTION.has(key) && Number(value) < 50)
    .map(([key]) => key);

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Micronutrients today" }),
      el("span", { class: "card__note", text: "% of daily value" })
    ),
    microGrid(summary.micronutrients, values),
    short.length && summary.meals_logged
      ? el("p", {
          class: "xsmall faint",
          text: `Running low on ${short.slice(0, 4).map(prettyKey).join(", ")}${short.length > 4 ? ` and ${short.length - 4} more` : ""}.`,
        })
      : null
  );
}

const CAUTION = new Set(["sodium_mg", "sugar_g"]);

const prettyKey = (key) =>
  key
    .replace(/_(g|mg|mcg)$/, "")
    .replace(/_/g, " ")
    .replace(/^vitamin /, "vit. ");

/* ---------------------------------------------------------------- Empty state */

/** First run. An empty dashboard full of zeroes is demoralising and teaches
 *  nothing, so this replaces it entirely until there is a single meal. */
function firstRun(summary) {
  const target = Number(summary.goal?.calories) || 2000;
  return el(
    "div",
    { class: "stack stack--lg" },
    el(
      "div",
      { class: "empty" },
      el("div", { class: "empty__icon" }, icon("target", { size: 28 })),
      el("h2", { class: "empty__title", text: "Nothing logged yet" }),
      el("p", {
        class: "empty__text",
        text: `Your goal is set to ${kcal(target)} kcal a day. Analyse a meal and it lands here — calories against the goal, the macro split, and how the day's micronutrients are shaping up.`,
      }),
      el(
        "div",
        { class: "row", style: { justifyContent: "center" } },
        el("a", { href: "/", class: "btn btn--primary" }, icon("camera", { size: 16 }), el("span", { text: "Analyse a meal" })),
        el("a", { href: "/settings", class: "btn btn--outline" }, icon("settings", { size: 16 }), el("span", { text: "Adjust goals" }))
      )
    ),
    el(
      "div",
      { class: "card stack stack--sm" },
      el("h3", { class: "card__title", text: "Your targets" }),
      macroList({}, { goals: goalMacros(summary.goal || {}) }),
      el(
        "div",
        { class: "row row--between small faint" },
        el("span", { text: "Calories" }),
        el("span", { text: `${kcal(target)} kcal` })
      )
    )
  );
}

function nudge() {
  return el(
    "div",
    { class: "panel panel--info" },
    el(
      "div",
      { class: "row row--between" },
      el(
        "div",
        { class: "row row--tight" },
        icon("info", { size: 16 }),
        el("strong", { text: "Nothing logged today yet" })
      ),
      el("a", { href: "/", class: "btn btn--sm btn--primary" }, icon("camera", { size: 15 }), el("span", { text: "Log one" }))
    )
  );
}
