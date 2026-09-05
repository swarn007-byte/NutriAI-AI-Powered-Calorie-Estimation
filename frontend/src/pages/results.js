/* Results — and history detail. One page, because a meal is a meal: the only
 * difference is whether it arrived in `state.meal` from a fresh analysis or has
 * to be fetched by id.
 *
 * The core interaction is the link between the photo and the list. Hovering or
 * focusing a row highlights its box on the plate and vice versa, which is what
 * makes a five-item thali readable at all.
 */

import { el, icon, kcal, grams, num, when, humanize, mount } from "../dom.js";
import { donut, macroSlices } from "../charts.js";
import { itemRow, macroList, microGrid, plateView, warningPanel } from "../components/meal.js";
import { api, ApiError } from "../api.js";
import { navigate } from "../router.js";
import { state, setMeal, ensureCatalog, isGuest, preference } from "../store.js";
import { backLink } from "../components/shell.js";
import { confirmAction, openSheet, toastError, toastOk } from "../toast.js";

export default async function results({ params }) {
  const page = el("div", { class: "page stack stack--lg results-page" });
  const wanted = params?.id;

  let meal = state.meal;
  if (wanted && meal?.meal_id !== wanted) meal = null;

  if (!meal) {
    if (!wanted) {
      // /results with nothing analysed — a bookmark or a refresh after a while.
      return el(
        "div",
        { class: "page" },
        el(
          "div",
          { class: "empty" },
          el("div", { class: "empty__icon" }, icon("camera", { size: 26 })),
          el("h1", { class: "empty__title", text: "No meal open" }),
          el("p", {
            class: "empty__text",
            text: "Analyse a photo to see a breakdown here, or pick one from your history.",
          }),
          el(
            "div",
            { class: "row", style: { justifyContent: "center" } },
            el("a", { href: "/", class: "btn btn--primary" }, icon("camera", { size: 16 }), el("span", { text: "Analyse a meal" })),
            el("a", { href: "/history", class: "btn btn--outline" }, icon("history", { size: 16 }), el("span", { text: "History" }))
          )
        )
      );
    }

    page.append(loadingBlock());
    api
      .meal(wanted)
      .then((fetched) => {
        if (!page.isConnected) return;
        render(page, fetched, { fromHistory: true });
      })
      .catch((error) => {
        if (!page.isConnected) return;
        page.replaceChildren(mealFailure(error));
      });
    return page;
  }

  render(page, meal, { fromHistory: Boolean(wanted) });
  return page;
}

function loadingBlock() {
  return el(
    "div",
    { class: "stack stack--lg", "aria-busy": "true" },
    el("div", { class: "skeleton", style: { height: "2.2rem", width: "min(20rem, 70%)" } }),
    el(
      "div",
      { class: "split" },
      el("div", { class: "skeleton", style: { aspectRatio: "4 / 3" } }),
      el(
        "div",
        { class: "stack stack--sm" },
        el("div", { class: "skeleton", style: { height: "9rem" } }),
        el("div", { class: "skeleton", style: { height: "3.4rem" } }),
        el("div", { class: "skeleton", style: { height: "3.4rem" } }),
        el("div", { class: "skeleton", style: { height: "3.4rem" } })
      )
    )
  );
}

function mealFailure(error) {
  const gone = error instanceof ApiError && error.status === 404;
  return el(
    "div",
    { class: "empty" },
    el("div", { class: "empty__icon" }, icon(gone ? "trash" : "alert", { size: 26 })),
    el("h1", { class: "empty__title", text: gone ? "That meal is gone" : "Couldn't load that meal" }),
    el("p", { class: "empty__text", text: gone ? "It was deleted, or it belongs to another account." : error?.message || "" }),
    el("a", { href: "/history", class: "btn btn--primary" }, icon("history", { size: 16 }), el("span", { text: "Back to history" }))
  );
}

/* -------------------------------------------------------------------- Render */

function render(page, meal, { fromHistory }) {
  /** Re-render in place after a correction. The server returns the whole meal,
   *  so there is nothing to reconcile — replace and redraw. */
  const update = (next) => {
    // Only the *active* meal is cached; a history item stays local to this view.
    if (state.meal?.meal_id === next.meal_id || !fromHistory) setMeal(next);
    render(page, next, { fromHistory });
  };

  const items = meal.items || [];
  const totals = meal.totals || {};
  const lowCount = items.filter((item) => item.low_confidence && !item.user_corrected).length;

  /* -------------------------------------------------------- Photo ↔ list link */

  const plate = plateView(meal, {
    onHover: (id) => highlightRow(id),
    onSelect: (id) => {
      const row = rowFor(id);
      row?.click();
      row?.scrollIntoView({ block: "nearest", behavior: "smooth" });
    },
  });

  const list = el("div", { class: "items" });
  const rowNodes = new Map();

  const rowFor = (id) => rowNodes.get(id);

  function highlightRow(id) {
    for (const [key, node] of rowNodes) node.classList.toggle("is-active", key === id);
  }

  items.forEach((item, index) => {
    const node = itemRow(item, index, {
      onHover: (id) => {
        highlightRow(id);
        plate.highlight(id);
      },
      onToggle: (id, open) => {
        if (open) plate.highlight(id);
      },
      onCorrect: (target) => openCorrection(meal, target, update),
    });
    rowNodes.set(item.id, node);
    list.appendChild(node);
  });

  /* ------------------------------------------------------------------ Header */

  const head = el(
    "div",
    { class: "stack stack--sm results__head" },
    el(
      "div",
      { class: "row row--between" },
      fromHistory ? backLink("/history", "History") : el("span", { class: "eyebrow", text: "Result" }),
      el(
        "div",
        { class: "row row--tight" },
        el("span", { class: "chip", "data-tip": new Date(meal.captured_at).toLocaleString() }, icon("clock", { size: 13 }), el("span", { text: when(meal.captured_at) })),
        meal.engine
          ? el("span", {
              class: ["chip", meal.engine === "full" ? "chip--brand" : ""],
              text: meal.engine === "full" ? "Model stack" : "Estimator",
              "data-tip": meal.engine === "full" ? "Analysed with the trained models" : "Analysed with the built-in estimator",
            })
          : null
      )
    ),
    el("h1", { class: "results__title", text: mealTitle(items) }),
    meal.notes ? el("p", { class: "muted small", text: `“${meal.notes}”` }) : null
  );

  /* ------------------------------------------------------------------ Totals */

  const slices = macroSlices(totals);
  const totalsCard = el(
    "div",
    { class: "card card--pad-lg results__totals" },
    el(
      "div",
      { class: "totals" },
      el("div", { class: "totals__ring" }, donut(slices, { total: totals.calories, label: "KCAL", size: 168 })),
      el(
        "div",
        { class: "stack stack--sm" },
        macroList(totals),
        /* No plate size here: the adjuster below states it in a chip, and repeating
         * it pushed this line to two, leaving a separator dot leading the second. */
        el(
          "div",
          { class: "row row--tight small faint" },
          el("span", { text: `${grams(totals.weight_g ?? sumWeight(items))} g total` }),
          el("span", { text: "·" }),
          el("span", { text: `${items.length} item${items.length === 1 ? "" : "s"}` })
        )
      )
    )
  );

  /* ------------------------------------------------- Low-confidence callout */

  const lowBanner =
    lowCount > 0
      ? el(
          "div",
          { class: "panel panel--warn" },
          el(
            "div",
            { class: "row row--between" },
            el(
              "div",
              { class: "row row--tight" },
              icon("alert", { size: 16 }),
              el("strong", {
                text:
                  lowCount === 1
                    ? "One item wasn't a confident match"
                    : `${lowCount} items weren't confident matches`,
              })
            ),
            el(
              "button",
              {
                class: "btn btn--sm btn--primary attention",
                onclick: () => {
                  const first = items.find((item) => item.low_confidence && !item.user_corrected);
                  if (first) openCorrection(meal, first, update);
                },
              },
              icon("edit", { size: 15 }),
              el("span", { text: "Review" })
            )
          ),
          el("p", {
            class: "small muted",
            style: { marginTop: "0.35rem" },
            text: "Correcting a label recomputes that item's nutrition immediately — and the correction is what a retrained model would learn from.",
          })
        )
      : null;

  /* -------------------------------------------------------------- Micros etc */

  const micros = el(
    "section",
    { class: "card stack stack--sm results__micros" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Micronutrients" }),
      el("span", { class: "card__note", text: "% of daily value" })
    ),
    microGrid(meal.micronutrients, meal.daily_values)
  );

  /* ----------------------------------------------------------- Plate rescale */

  const plateCard = plateAdjuster(meal, update);

  /* ---------------------------------------------------------------- Actions */

  const actions = el(
    "div",
    { class: "row results__actions" },
    el("a", { href: "/", class: "btn btn--primary" }, icon("camera", { size: 16 }), el("span", { text: "Analyse another" })),
    el("a", { href: "/today", class: "btn btn--outline" }, icon("target", { size: 16 }), el("span", { text: "See today" })),
    el("span", { class: "grow" }),
    deleteButton(meal)
  );

  /* --------------------------------------------------------------- Assemble */

  // mount(), not replaceChildren(): warningPanel and lowBanner are both absent
  // on a clean meal, and native replaceChildren stringifies a null child into a
  // literal "null" on the page.
  mount(
    page,
    head,
    warningPanel(meal.warnings),
    lowBanner,
    el(
      "div",
      { class: "split results__layout" },
      el("div", { class: "split__sticky stack stack--sm results__visual" }, plate, plateCard),
      el(
        "div",
        { class: "stack stack--sm results__details" },
        totalsCard,
        el(
          "section",
          { class: "section" },
          el(
            "div",
            { class: "section__head" },
            el("h2", { class: "section__title", text: "What's on the plate" }),
            el("span", { class: "section__note", text: "Tap an item for the geometry" })
          ),
          list
        )
      )
    ),
    micros,
    timingsCard(meal),
    actions
  );

  // Clear the highlight when the pointer leaves either surface, or the last
  // hovered item stays lit and reads as a selection.
  list.addEventListener("mouseleave", () => {
    highlightRow(null);
    plate.highlight(null);
  });
}

const sumWeight = (items) =>
  items.reduce((acc, item) => acc + (Number(item.estimated_weight_g) || 0), 0);

/** Name the meal after its two biggest items by calories — "Rice and Dal Tadka"
 *  is a better page heading than "Meal" or a timestamp. */
function mealTitle(items) {
  const named = [...items]
    .sort((a, b) => (b.calories || 0) - (a.calories || 0))
    .map((item) => item.display_name)
    .filter(Boolean);
  if (!named.length) return "Meal";
  if (named.length === 1) return named[0];
  const rest = named.length - 2;
  return `${named[0]} and ${named[1]}${rest > 0 ? ` +${rest} more` : ""}`;
}

/* ------------------------------------------------------------ Plate rescale */

/** Re-scaling after the fact is the highest-leverage correction in the app: the
 *  plate diameter is the pixel→cm factor, and volume scales with its cube. A
 *  26 cm guess on a 30 cm plate is a 53% weight error on every item at once. */
function plateAdjuster(meal, update) {
  const current = Number(meal.plate_diameter_cm) || Number(preference("plate_diameter_cm", 26));
  let value = current;

  const output = el("b", { text: `${num(value, 1)} cm` });
  const apply = el(
    "button",
    { class: "btn btn--sm btn--outline", disabled: true },
    icon("scale", { size: 15 }),
    el("span", { text: "Rescale" })
  );

  const slider = el("input", {
    type: "range",
    class: "range",
    min: "12",
    max: "45",
    step: "0.5",
    value: String(value),
    "aria-label": "Actual plate diameter in centimetres",
    oninput: (event) => {
      value = Number(event.target.value);
      output.textContent = `${num(value, 1)} cm`;
      const changed = Math.abs(value - current) >= 0.5;
      apply.disabled = !changed;
      delta.textContent = changed
        ? `Weights ×${num((value / current) ** 3, 2)}`
        : "Matches what was used";
    },
  });

  const delta = el("span", { class: "small faint", text: "Matches what was used" });

  apply.addEventListener("click", async () => {
    apply.setAttribute("aria-busy", "true");
    try {
      const next = await api.adjustPlate(meal.meal_id, value);
      toastOk(`Rescaled for a ${num(value, 1)} cm plate.`);
      update(next);
    } catch (error) {
      toastError(error?.message || "Couldn't rescale that meal.");
    } finally {
      apply.removeAttribute("aria-busy");
    }
  });

  return el(
    "details",
    { class: "card" },
    el(
      "summary",
      { class: "row row--between", style: { cursor: "pointer", listStyle: "none" } },
      el(
        "span",
        { class: "row row--tight" },
        icon("ruler", { size: 16 }),
        el("span", { class: "settings-row__title", text: "Wrong plate size?" })
      ),
      el("span", { class: "chip" }, output)
    ),
    el(
      "div",
      { class: "stack stack--sm", style: { marginTop: "var(--space-2xs)" } },
      el("p", {
        class: "small muted",
        text: "Everything is measured against the plate. Set its real width and every item that you have not corrected yourself is rescaled — volume by the cube of the change.",
      }),
      slider,
      el("div", { class: "row row--between" }, delta, apply)
    )
  );
}

/* ---------------------------------------------------------------- Deletion */

function deleteButton(meal) {
  const button = el(
    "button",
    { class: "btn btn--ghost btn--sm" },
    icon("trash", { size: 15 }),
    el("span", { text: "Delete" })
  );

  button.addEventListener("click", async () => {
    const ok = await confirmAction({
      title: "Delete this meal?",
      message: "The photo is removed from disk and the entry leaves your history and today's total. This cannot be undone.",
      confirmLabel: "Delete",
      danger: true,
    });
    if (!ok) return;
    try {
      await api.deleteMeal(meal.meal_id);
      if (state.meal?.meal_id === meal.meal_id) setMeal(null);
      toastOk("Meal deleted.");
      navigate("/history", { replace: true });
    } catch (error) {
      toastError(error?.message || "Couldn't delete that meal.");
    }
  });

  return button;
}

/* ----------------------------------------------------------------- Timings */

/** Per-stage cost, straight from the response. This is here because the whole
 *  premise of the app is that the pipeline is inspectable — a user who wonders
 *  why a result took four seconds can see which stage spent them. */
function timingsCard(meal) {
  const timings = meal.timings_ms || {};
  const keys = Object.keys(timings);
  if (!keys.length) return null;

  const total = keys.reduce((acc, key) => acc + (Number(timings[key]) || 0), 0);
  const versions = meal.model_versions || {};

  return el(
    "details",
    { class: "card" },
    el(
      "summary",
      { class: "row row--between", style: { cursor: "pointer", listStyle: "none" } },
      el(
        "span",
        { class: "row row--tight" },
        icon("layers", { size: 16 }),
        el("span", { class: "settings-row__title", text: "Pipeline detail" })
      ),
      el("span", { class: "chip", text: `${num(total)} ms` })
    ),
    el(
      "div",
      { class: "table-wrap", style: { marginTop: "var(--space-2xs)" } },
      el(
        "table",
        { class: "table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", { text: "Stage" }),
            el("th", { text: "Model" }),
            el("th", { class: "num", text: "Time" })
          )
        ),
        el(
          "tbody",
          {},
          keys.map((key) =>
            el(
              "tr",
              {},
              el("td", { text: humanize(key) }),
              el("td", { class: "small faint", text: versions[key] || "—" }),
              el("td", { class: "num", text: `${num(timings[key])} ms` })
            )
          )
        )
      )
    )
  );
}

/* ---------------------------------------------------------- Correction sheet */

/** The correction flow. Three ways to fix an item, in order of how often they
 *  are the right one: pick one of the model's own runners-up, search the food
 *  catalogue, or override the weight directly.
 */
async function openCorrection(meal, item, update) {
  const alternatives = Array.isArray(item.alternatives) ? item.alternatives : [];

  let label = null;
  let weight = Number(item.estimated_weight_g) || 0;

  const suggestions = el("div", { class: "suggestions" });
  const searchResults = el("div", { class: "suggestions" });

  const save = el(
    "button",
    { class: "btn btn--primary", disabled: true },
    icon("check", { size: 16 }),
    el("span", { text: "Save correction" })
  );

  const markDirty = () => {
    const changed = label !== null || Math.abs(weight - (Number(item.estimated_weight_g) || 0)) >= 0.5;
    save.disabled = !changed;
  };

  function pick(nextLabel, node) {
    label = nextLabel;
    for (const button of [...suggestions.children, ...searchResults.children]) {
      button.setAttribute("aria-pressed", String(button === node));
    }
    markDirty();
  }

  /* --- The model's own alternatives, which is usually where the answer is --- */

  if (alternatives.length) {
    suggestions.append(
      ...alternatives.map((alt) => {
        const button = el(
          "button",
          { class: "suggestion", "aria-pressed": "false" },
          el("span", { class: "suggestion__name", text: alt.display_name || humanize(alt.label) }),
          alt.confidence !== undefined
            ? el("span", { class: "suggestion__conf", text: `${Math.round(alt.confidence * 100)}%` })
            : null
        );
        button.addEventListener("click", () => pick(alt.label, button));
        return button;
      })
    );
  }

  /* ------------------------------- Catalogue search, for everything else --- */

  const search = el("input", {
    class: "input",
    type: "search",
    placeholder: "Search all foods…",
    "aria-label": "Search the food catalogue",
  });

  let catalog = state.catalog;
  ensureCatalog()
    .then((loaded) => {
      catalog = loaded;
    })
    .catch(() => {
      searchResults.replaceChildren(
        el("p", { class: "small muted", text: "The food list couldn't be loaded." })
      );
    });

  search.addEventListener("input", () => {
    const query = search.value.trim().toLowerCase();
    if (!query || !catalog) {
      searchResults.replaceChildren();
      return;
    }
    const entries = catalog.filter((entry) => {
      const name = String(entry.display_name || "").toLowerCase();
      return name.includes(query) || String(entry.label || "").includes(query.replace(/\s+/g, "_"));
    });

    if (!entries.length) {
      searchResults.replaceChildren(
        el("p", { class: "small muted", text: `Nothing matching “${search.value.trim()}”.` })
      );
      return;
    }

    searchResults.replaceChildren(
      ...entries.slice(0, 8).map((entry) => {
        const button = el(
          "button",
          { class: "suggestion", "aria-pressed": "false" },
          el(
            "span",
            { class: "suggestion__name" },
            el("span", { text: entry.display_name || humanize(entry.label) }),
            el("span", { class: "xsmall faint", style: { display: "block" }, text: humanize(entry.category || "") })
          ),
          el("span", { class: "suggestion__conf", text: `${kcal(entry.kcal_per_100g)} /100g` })
        );
        button.addEventListener("click", () => pick(entry.label, button));
        return button;
      })
    );
  });

  /* ------------------------------------------------------ Weight override --- */

  const weightOut = el("b", { text: `${grams(weight)} g` });
  const weightSlider = el("input", {
    type: "range",
    class: "range",
    min: "0",
    // A 3 kg ceiling matches the API bound; the practical range is far below it,
    // so the scale is anchored on this item's own estimate ×3 to stay usable.
    max: String(Math.max(300, Math.round(weight * 3))),
    step: "5",
    value: String(Math.round(weight)),
    "aria-label": "Corrected weight in grams",
    oninput: (event) => {
      weight = Number(event.target.value);
      weightOut.textContent = `${grams(weight)} g`;
      markDirty();
    },
  });

  /* -------------------------------------------------------------- Assemble --- */

  const body = el(
    "div",
    { class: "stack stack--sm" },
    el(
      "div",
      { class: "row row--between" },
      el(
        "div",
        {},
        el("div", { class: "settings-row__title", text: item.display_name }),
        el("div", {
          class: "settings-row__note",
          text: item.detected_label && item.detected_label !== item.classified_label
            ? `Detected as ${humanize(item.detected_label)}`
            : `${grams(item.estimated_weight_g)} g · ${kcal(item.calories)} kcal`,
        })
      ),
      item.confidence !== undefined && !item.user_corrected
        ? el("span", { class: "chip chip--warn", text: `${Math.round(item.confidence * 100)}% sure` })
        : null
    ),
    alternatives.length
      ? el(
          "div",
          { class: "stack stack--sm" },
          el("span", { class: "eyebrow", text: "Did you mean" }),
          suggestions
        )
      : null,
    el(
      "div",
      { class: "field" },
      el("span", { class: "field__label", text: "Or find the right food" }),
      search,
      searchResults
    ),
    el(
      "div",
      { class: "field" },
      el(
        "span",
        { class: "row row--between" },
        el("span", { class: "field__label", text: "Weight" }),
        el("span", { class: "chip" }, weightOut)
      ),
      weightSlider,
      el("span", {
        class: "field__hint",
        text: "Only change this if you know the portion. It replaces the measured estimate outright.",
      })
    )
  );

  const cancel = el("button", { class: "btn btn--ghost", text: "Cancel" });
  const { close } = openSheet({
    title: "Fix this item",
    body,
    footer: [cancel, save],
  });
  cancel.addEventListener("click", () => close());

  save.addEventListener("click", async () => {
    save.setAttribute("aria-busy", "true");
    const payload = {};
    if (label !== null) payload.classified_label = label;
    if (Math.abs(weight - (Number(item.estimated_weight_g) || 0)) >= 0.5) {
      payload.estimated_weight_g = weight;
    }
    try {
      const next = await api.correctItem(meal.meal_id, item.id, payload);
      close();
      toastOk("Corrected — nutrition recomputed.");
      update(next);
    } catch (error) {
      toastError(error?.message || "That correction didn't save.");
      save.removeAttribute("aria-busy");
    }
  });

  // Guests can correct freely; the nudge is about not losing the record later.
  if (isGuest()) {
    body.appendChild(
      el(
        "p",
        { class: "xsmall faint" },
        "You're browsing as a guest. ",
        el("a", { href: "/auth", text: "Create an account" }),
        " to keep this meal and its corrections."
      )
    );
  }
}
