/* Meal presentation components, shared by the results page and the history
 * detail view (which are the same page — a meal is a meal).
 */

import { el, svg, icon, kcal, grams, num, micro, pct, nutrientLabel, clamp } from "../dom.js";
import { macroBar, MACROS } from "../charts.js";

/* --------------------------------------------------------- Annotated photo */

/** The photo with an SVG bounding-box overlay.
 *
 *  The SVG uses the *image's own* pixel dimensions as its viewBox, so the
 *  fractional bboxes the API returns map to exact coordinates and the whole
 *  thing scales with the image at any viewport width. No resize observer.
 *
 *  `onHover`/`onSelect` link it to the item list so hovering a row highlights
 *  the box and vice versa — the single most useful affordance on this screen,
 *  since a plate of five curries is otherwise unreadable.
 */
export function plateView(meal, { onHover, onSelect } = {}) {
  const width = meal.image_width || 1000;
  const height = meal.image_height || 1000;

  const overlay = svg("svg", {
    class: "plate__overlay",
    viewBox: `0 0 ${width} ${height}`,
    preserveAspectRatio: "none",
    role: "group",
    "aria-label": "Detected food regions",
  });

  const shapes = new Map();

  meal.items.forEach((item, index) => {
    if (!item.bbox) return;
    const x = clamp(item.bbox.x, 0, 1) * width;
    const y = clamp(item.bbox.y, 0, 1) * height;
    const w = Math.max(4, Math.min(width - x, clamp(item.bbox.w, 0, 1) * width));
    const h = Math.max(4, Math.min(height - y, clamp(item.bbox.h, 0, 1) * height));
    const low = item.low_confidence && !item.user_corrected;

    const group = svg("g", { class: "bbox-group", dataset: { item: item.id } });

    const rect = svg("rect", {
      class: ["bbox", low && "bbox--low"],
      x,
      y,
      width: w,
      height: h,
      rx: Math.min(14, w / 6, h / 6),
      role: "button",
      "aria-label": `${index + 1}. ${item.display_name}`,
    });

    // The label sits above the box, or inside it when the box is near the top
    // edge — otherwise it is clipped off the canvas on top-of-frame items.
    const rawLabel = `${index + 1}. ${item.display_name}`;
    const fontSize = Math.max(13, Math.round(width * 0.019));
    const padX = fontSize * 0.45;
    const maxChars = Math.max(10, Math.floor((width - padX * 2) / (fontSize * 0.53)));
    const labelText = rawLabel.length > maxChars ? `${rawLabel.slice(0, maxChars - 1)}…` : rawLabel;
    const chipW = labelText.length * fontSize * 0.53 + padX * 2;
    const chipH = fontSize * 1.7;
    const above = y > chipH + 6;
    const chipY = above ? y - chipH - 4 : y + 4;
    const chipX = Math.max(0, Math.min(x, width - chipW));

    group.append(
      rect,
      svg("rect", {
        class: ["bbox-label__bg", low && "bbox-label__bg--low"],
        x: chipX,
        y: chipY,
        width: Math.min(chipW, width - chipX),
        height: chipH,
        rx: chipH / 2.6,
      }),
      svg("text", {
        class: "bbox-label",
        x: chipX + padX,
        y: chipY + chipH * 0.68,
        "font-size": fontSize,
        text: labelText,
      })
    );

    if (onHover) {
      group.addEventListener("mouseenter", () => onHover(item.id));
      group.addEventListener("mouseleave", () => onHover(null));
    }
    if (onSelect) {
      rect.addEventListener("click", () => onSelect(item.id));
      rect.setAttribute("tabindex", "0");
      rect.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(item.id);
        }
      });
    }

    overlay.appendChild(group);
    shapes.set(item.id, group);
  });

  const node = el(
    "figure",
    { class: "plate" },
    meal.image_url
      ? el("img", {
          src: meal.image_url,
          alt: `Analysed meal photo containing ${meal.items.map((i) => i.display_name).join(", ") || "no identified items"}`,
          width,
          height,
          loading: "eager",
          decoding: "async",
        })
      : el("div", { class: "empty", style: { aspectRatio: "4 / 3" } }, el("span", { text: "Image unavailable" })),
    overlay
  );

  /** Called by the results page when the active item changes. */
  node.highlight = (itemId) => {
    for (const [id, group] of shapes) {
      group.querySelector(".bbox")?.classList.toggle("is-active", id === itemId);
    }
  };

  return node;
}

/* ---------------------------------------------------------------- Item rows */

/** One food item. Collapsed it shows name, weight and calories; expanded it
 *  shows the macro split and how the weight was arrived at.
 *
 *  The geometry disclosure is deliberate. A number like "182 g" invites
 *  "how do you know?", and the honest answer — a mean depth prior over a
 *  measured footprint — is more trustworthy than silence.
 */
export function itemRow(item, index, { onToggle, onHover, onCorrect } = {}) {
  const low = item.low_confidence && !item.user_corrected;
  const geometry = item.geometry || {};

  const detail = el(
    "div",
    { class: "item__detail" },
    el(
      "div",
      {},
      el(
        "div",
        { class: "item__detail-inner" },
        macroBar(item),
        el(
          "div",
          { class: "kv" },
          ...MACROS.map((macro) =>
            el(
              "div",
              {},
              el("div", { class: "kv__k", text: macro.label }),
              el("div", { class: "kv__v", style: { color: macro.color }, text: `${grams(item[macro.key])} g` })
            )
          ),
          item.nutrients?.fiber_g
            ? el(
                "div",
                {},
                el("div", { class: "kv__k", text: "Fibre" }),
                el("div", { class: "kv__v", text: `${grams(item.nutrients.fiber_g)} g` })
              )
            : null,
          el(
            "div",
            {},
            el("div", { class: "kv__k", text: "Volume" }),
            el("div", {
              class: "kv__v",
              text: item.estimated_volume_ml ? `${num(item.estimated_volume_ml)} ml` : "—",
            })
          ),
          item.piece_count && item.piece_weight_g
            ? el(
                "div",
                {},
                el("div", { class: "kv__k", text: "Per piece" }),
                el("div", { class: "kv__v", text: `${grams(item.piece_weight_g)} g` })
              )
            : null,
          geometry.area_cm2
            ? el(
                "div",
                {},
                el("div", { class: "kv__k", text: "Footprint" }),
                el("div", { class: "kv__v", text: `${num(geometry.area_cm2)} cm²` })
              )
            : null,
          geometry.mean_height_cm
            ? el(
                "div",
                {},
                el("div", { class: "kv__k", text: "Mean depth" }),
                el("div", { class: "kv__v", text: `${num(geometry.mean_height_cm, 1)} cm` })
              )
            : null
        ),
        el(
          "div",
          { class: "row row--tight" },
          item.nutrition_source
            ? el("span", { class: "chip", text: item.nutrition_source })
            : null,
          geometry.method ? el("span", { class: "chip", text: methodLabel(geometry.method) }) : null,
          geometry.clamped
            ? el("span", {
                class: "chip chip--warn",
                text: "Portion capped",
                "data-tip": "The estimate fell outside a plausible serving range and was pulled back to the nearest bound.",
              })
            : null
        ),
        onCorrect
          ? el(
              "div",
              { class: "row row--tight" },
              el(
                "button",
                {
                  class: ["btn", "btn--sm", low ? "btn--primary" : "btn--outline", low ? "attention" : ""],
                  onclick: (event) => {
                    event.stopPropagation();
                    onCorrect(item);
                  },
                },
                icon("edit", { size: 15 }),
                el("span", { text: low ? "Not right? Fix it" : "Adjust" })
              )
            )
          : null
      )
    )
  );

  const node = el(
    "div",
    {
      class: ["item", low && "item--low"],
      dataset: { item: item.id },
      role: "button",
      tabindex: "0",
      "aria-expanded": "false",
    },
    el("span", { class: "item__index", text: String(index + 1) }),
    el(
      "div",
      { class: "item__body" },
      el(
        "div",
        { class: "item__name" },
        el("span", { class: "truncate", text: item.display_name }),
        item.user_corrected
          ? el("span", { class: "chip chip--brand", "data-tip": "You set this", text: "Yours" })
          : null,
        low ? el("span", { class: "chip chip--warn", text: "Unsure" }) : null
      ),
      el(
        "div",
        { class: "item__meta" },
        // A count is more meaningful than the gram figure it implies, so it leads
        // when there is one — "4 pieces · 260 g" reads as an observation, while
        // "260 g" alone reads as a guess the user has to take on trust.
        item.piece_count
          ? el("span", { text: `${item.piece_count} ${item.piece_count === 1 ? "piece" : "pieces"}` })
          : null,
        item.piece_count ? el("span", { class: "faint", text: "·" }) : null,
        el("span", { text: `${grams(item.estimated_weight_g)} g` }),
        el("span", { class: "faint", text: "·" }),
        el("span", { text: `P ${grams(item.protein_g)} · C ${grams(item.carbs_g)} · F ${grams(item.fat_g)}` }),
        !item.user_corrected && item.confidence
          ? el("span", { class: "faint", text: `· ${pct(item.confidence * 100)} sure` })
          : null
      )
    ),
    el(
      "div",
      { class: "item__kcal" },
      el("b", { text: kcal(item.calories) }),
      el("span", { text: "kcal" })
    ),
    detail
  );

  const toggle = () => {
    const open = node.classList.toggle("is-open");
    node.setAttribute("aria-expanded", String(open));
    onToggle?.(item.id, open);
  };

  node.addEventListener("click", toggle);
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      toggle();
    }
  });
  if (onHover) {
    node.addEventListener("mouseenter", () => onHover(item.id));
    node.addEventListener("mouseleave", () => onHover(null));
  }

  return node;
}

const METHOD_LABELS = {
  "midas+geometry": "Depth model + geometry",
  "shape-prior+geometry": "Shape prior + geometry",
  "fallback-portion": "Typical portion",
  "piece-count": "Counted by the piece",
  "nominal-portion": "Standard serving",
  "user-corrected": "Your correction",
  "plate-recalibrated": "Re-measured for plate size",
};

const methodLabel = (method) => METHOD_LABELS[method] || String(method);

/* ------------------------------------------------------- Micronutrient grid */

/** %DV bars for every tracked micronutrient.
 *
 *  Sodium and sugar are inverted in meaning — high is bad — so they are tinted
 *  amber past 100% rather than green. Reading "128% sodium" in the same colour
 *  as "128% vitamin C" would be actively misleading.
 */
const CAUTION_KEYS = new Set(["sodium_mg", "sugar_g"]);

export function microGrid(micronutrients, dailyValues) {
  const keys = Object.keys(dailyValues || {});
  if (!keys.length) return el("p", { class: "muted small", text: "No micronutrient data." });

  return el(
    "div",
    { class: "micros" },
    keys.map((key) => {
      const { name, unit } = nutrientLabel(key);
      const percent = Number(dailyValues[key]) || 0;
      const amount = Number(micronutrients?.[key]) || 0;
      const caution = CAUTION_KEYS.has(key) && percent > 100;
      const colour = caution ? "var(--warn)" : percent >= 25 ? "var(--brand)" : "var(--text-faint)";

      return el(
        "div",
        { class: "micro" },
        el(
          "div",
          { class: "micro__top" },
          el("span", { class: "micro__name", text: name }),
          el(
            "span",
            { class: "micro__val" },
            `${micro(amount)} ${unit}`,
            el("b", {
              class: "micro__pct",
              style: { color: colour, marginLeft: "0.4rem" },
              text: pct(percent),
            })
          )
        ),
        el(
          "div",
          { class: "meter meter--thin", style: { "--meter-fill": colour } },
          el("div", {
            class: ["meter__fill", percent > 100 && "meter__fill--over"],
            style: { width: `${clamp(percent, 0, 100)}%` },
          })
        )
      );
    })
  );
}

/* ------------------------------------------------------------ Macro summary */

export function macroList(source, { goals } = {}) {
  const total = MACROS.reduce(
    (acc, macro) => acc + (Number(source?.[macro.key]) || 0) * macro.kcalPerGram,
    0
  );

  return el(
    "div",
    { class: "macrolist" },
    MACROS.map((macro) => {
      const value = Number(source?.[macro.key]) || 0;
      const energy = value * macro.kcalPerGram;
      const share = total > 0 ? (energy / total) * 100 : 0;
      const goal = goals?.[macro.key];

      return el(
        "div",
        { class: "macrorow" },
        el("span", { class: "dot", style: { "--c": macro.color } }),
        el(
          "div",
          { class: "grow" },
          el("div", { class: "macrorow__name", text: macro.label }),
          el(
            "div",
            { class: "meter meter--thin", style: { "--meter-fill": macro.color } },
            el("div", {
              class: "meter__fill",
              style: { width: `${clamp(goal ? (value / goal) * 100 : share, 0, 100)}%` },
            })
          )
        ),
        el(
          "span",
          { class: "macrorow__val" },
          el("b", { text: `${grams(value)} g` }),
          goal ? ` / ${grams(goal)} g` : ` · ${pct(share)}`
        )
      );
    })
  );
}

/* ------------------------------------------------------------------ Warnings */

/** Pipeline warnings, verbatim from the backend. These are the honest caveats
 *  (no plate detected, portion clamped, coarse label) and hiding them would
 *  make the numbers look more certain than they are. */
export function warningPanel(warnings) {
  if (!warnings?.length) return null;
  return el(
    "div",
    { class: "panel panel--warn" },
    el(
      "div",
      { class: "row row--tight", style: { marginBottom: "0.35rem" } },
      icon("alert", { size: 16 }),
      el("strong", { text: warnings.length === 1 ? "One thing to know" : `${warnings.length} things to know` })
    ),
    el(
      "ul",
      { class: "stack stack--sm small muted" },
      warnings.map((warning) => el("li", { text: warning }))
    )
  );
}
