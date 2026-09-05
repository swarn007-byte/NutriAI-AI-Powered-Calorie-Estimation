/* Method — how a number gets made, in enough detail to argue with.
 *
 * This page exists because the app's core claim ("182 g of paneer butter
 * masala") is unverifiable by eye. The only honest response is to publish the
 * chain of reasoning, the priors, and the size of the error bar. Every constant
 * quoted here is the real one from the backend, not a rounded illustration.
 */

import { el, icon, num } from "../dom.js";
import { state } from "../store.js";

export default function method() {
  const page = el("div", { class: "page stack stack--lg" });
  const health = state.health;
  const limits = health?.limits || {};

  page.append(
    el(
      "div",
      { class: "stack stack--sm" },
      el("span", { class: "eyebrow", text: "Transparency" }),
      el("h1", { style: { fontSize: "var(--step-4)", maxWidth: "24ch" }, text: "How the estimate is made" }),
      el("p", {
        class: "hero__lede",
        style: { maxWidth: "58ch" },
        text: "One photo, five stages, and a measurement you can check. Nothing here is a look-up on the filename or a guess from the dish name — the weight comes out of the pixels, and where a prior is used instead, this page says which one.",
      })
    ),

    el(
      "div",
      { class: "panel panel--info" },
      el(
        "div",
        { class: "row row--tight", style: { marginBottom: "0.35rem" } },
        icon("info", { size: 16 }),
        el("strong", { text: "The short version" })
      ),
      el("p", {
        class: "small muted",
        text: "The plate's width sets the scale. The item's outline gives its footprint in cm². A depth field gives the shape of the mound, and a per-category depth prior gives its height. Footprint × height is volume; volume × density is grams; grams × per-100 g nutrition is the number you see.",
      })
    ),

    stagesCard(),
    formulaCard(),
    scaleCard(),
    boundsCard(),
    accuracyCard(),
    engineCard(health),
    limitsCard(limits),
    sourcesCard()
  );

  return page;
}

/* ------------------------------------------------------------------ Stages */

const STAGES = [
  {
    n: 1,
    title: "Find every item",
    icon: "layers",
    text: "The plate is located first — a bright, low-chroma, near-centre disc with its holes filled, because the food sitting on the plate punches holes in that mask and the scale must be measured from the whole disc. Then each food region is separated. A thali is five items, not one average.",
    real: "YOLOv8-seg when weights are installed; otherwise a plate-aware colour/texture segmenter.",
  },
  {
    n: 2,
    title: "Read the depth",
    icon: "cube",
    text: "A monocular depth model turns the flat photo into a relative height field. Relative is the operative word: it says this part is higher than that part, not how many centimetres. The rim of each blob anchors zero height at the 12th percentile of depth inside it, and the 98th percentile sets the span.",
    real: "MiDaS v3 when available; otherwise the Euclidean distance transform of the mask, which gives the classic food-mound dome.",
  },
  {
    n: 3,
    title: "Measure the portion",
    icon: "ruler",
    text: "The elevation field is rescaled to unit mean and summed — a Riemann sum over the pixels. That decouples volume from how peaked the shape happens to be, so a dome and a flat slab with the same footprint and the same mean depth hold the same volume, which is the physically correct answer.",
    real: "Plain geometry. No model, no learned parameters, identical on every install.",
  },
  {
    n: 4,
    title: "Name the dish",
    icon: "brain",
    text: "The crop is classified against a food catalogue rather than left as a coarse label, because the label chooses the density and the nutrition row. “Curry” and “paneer butter masala” are about 90 kcal apart per 100 g — the naming error dominates the geometry error when it goes wrong.",
    real: "EfficientNet-B3 when a checkpoint exists; otherwise a colour/texture signature prior over the same 42 classes.",
  },
  {
    n: 5,
    title: "Cost the nutrition",
    icon: "scale",
    text: "The dish is looked up per 100 g and scaled linearly to the estimated weight. Calories, three macros, fibre and thirteen micronutrients all come from the same row, so they are internally consistent even when the weight is wrong.",
    real: "IFCT 2017 for Indian dishes, USDA FoodData Central for the rest.",
  },
];

function stagesCard() {
  return el(
    "section",
    { class: "stack stack--sm" },
    el("h2", { class: "card__title", style: { fontSize: "var(--step-2)" }, text: "The five stages" }),
    el(
      "div",
      { class: "stack stack--sm" },
      STAGES.map((stage) =>
        el(
          "article",
          { class: "card stack stack--xs" },
          el(
            "div",
            { class: "row row--tight" },
            el("span", { class: "stage__mark", text: String(stage.n) }),
            icon(stage.icon, { size: 17 }),
            el("h3", { class: "card__title", text: stage.title })
          ),
          el("p", { class: "small muted", text: stage.text }),
          el(
            "div",
            { class: "row row--tight xsmall faint" },
            el("span", { class: "chip", text: "What runs" }),
            el("span", { text: stage.real })
          )
        )
      )
    )
  );
}

/* ----------------------------------------------------------------- Formula */

function formulaCard() {
  return el(
    "section",
    { class: "card stack stack--sm" },
    el("h2", { class: "card__title", text: "The arithmetic, in full" }),
    el("div", {
      class: "formula",
      text: [
        "cm_per_px   = plate_diameter_cm / (2 · plate_radius_px)",
        "area_cm²    = mask_pixels · cm_per_px²",
        "",
        "mean_depth  = depth_prior[category]",
        "              · clip((area_cm² / 60)^0.22, 0.65, 1.6)",
        "              clipped to [0.2 cm, 6.0 cm]",
        "",
        "h(px)       = elevation(px) · mean_depth · n_px / Σ elevation",
        "volume_ml   = Σ h(px) · cm_per_px²          (1 cm³ = 1 ml)",
        "",
        "weight_g    = volume_ml · density[dish]",
        "              clipped to the category's served range",
        "",
        "nutrient    = per_100g[dish] · weight_g / 100",
      ].join("\n"),
    }),
    el(
      "div",
      { class: "prose" },
      el("p", {
        text: "Two of those lines carry the uncertainty. The first is cm_per_px, which is only as good as the plate width you supply. The second is mean_depth, which is a prior rather than a measurement — a single photograph contains no absolute scale in the vertical direction, so something has to stand in for it.",
      }),
      el("p", {
        text: "Everything else is arithmetic that would give the same answer on any install, with or without trained weights.",
      })
    )
  );
}

/* ------------------------------------------------------------------- Scale */

function scaleCard() {
  const rows = [
    ["12–16 cm", "Katori, small bowl, saucer"],
    ["18–22 cm", "Side plate, cereal bowl"],
    ["24–28 cm", "Dinner plate — the default is 26 cm"],
    ["30–35 cm", "Large thali, serving plate"],
    ["36–45 cm", "Sharing platter, tray"],
  ];

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Why the plate width matters so much" }),
      el("span", { class: "card__note", text: "The one input you control" })
    ),
    el(
      "div",
      { class: "prose" },
      el("p", {
        text: "Area scales with the square of cm_per_px and volume adds the depth term, so an error in the plate width propagates roughly as its cube. Get the width 10% wrong and every weight on the plate is out by about a third — in the same direction, which is why the totals can look plausible while each item is wrong.",
      }),
      el("p", {
        html: "That is also why it is <strong>correctable after the fact</strong>: changing the plate width on a finished result rescales every item by <code>(new / old)³</code> rather than re-running the models.",
      })
    ),
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        { class: "table" },
        el("thead", {}, el("tr", {}, el("th", { text: "Width" }), el("th", { text: "Typical vessel" }))),
        el(
          "tbody",
          {},
          rows.map(([width, what]) => el("tr", {}, el("td", { class: "nowrap", text: width }), el("td", { class: "muted", text: what })))
        )
      )
    )
  );
}

/* ------------------------------------------------------------------ Bounds */

/* The real table from depth.py:79. Publishing it is the point: a clamped
 * estimate is reported as clamped, and this is what it was clamped to. */
const BOUNDS = [
  ["Rice", 50, 420, 2.2],
  ["Curry", 45, 340, 2.4],
  ["Dal", 50, 360, 2.0],
  ["Grain", 40, 380, 2.2],
  ["Bread", 18, 220, 0.6],
  ["Dry sabzi", 35, 300, 2.2],
  ["Protein", 30, 330, 2.2],
  ["Fried", 20, 260, 2.0],
  ["Dairy", 30, 300, 2.0],
  ["Dessert", 25, 260, 2.4],
  ["Salad", 20, 240, 2.8],
  ["Fruit", 40, 400, 2.6],
  ["Steamed", 30, 300, 1.7],
  ["Condiment", 8, 120, 1.5],
  ["Unknown", 25, 350, 2.4],
];

function boundsCard() {
  return el(
    "details",
    { class: "card stack stack--sm" },
    el(
      "summary",
      { class: "row row--between", style: { cursor: "pointer" } },
      el("h2", { class: "card__title", text: "Priors and plausible ranges" }),
      icon("chevronDown", { size: 17 })
    ),
    el("p", {
      class: "small muted",
      text: "Each category carries a mean served depth and a range of plausible served weights. An estimate outside its range is pulled to the nearest bound and labelled “portion capped” on the result, rather than being reported as if it were measured.",
    }),
    el(
      "div",
      { class: "table-wrap" },
      el(
        "table",
        { class: "table" },
        el(
          "thead",
          {},
          el(
            "tr",
            {},
            el("th", { text: "Category" }),
            el("th", { text: "Served weight" }),
            el("th", { text: "Mean depth" })
          )
        ),
        el(
          "tbody",
          {},
          BOUNDS.map(([label, low, high, depth]) =>
            el(
              "tr",
              {},
              el("td", { text: label }),
              el("td", { class: "nowrap", text: `${num(low)}–${num(high)} g` }),
              el("td", { class: "faint nowrap", text: `${num(depth, 1)} cm` })
            )
          )
        )
      )
    ),
    el("p", {
      class: "xsmall faint",
      text: "Depth is adjusted for footprint before use: a larger helping is both wider and slightly deeper, so the prior is multiplied by (area / 60 cm²)^0.22, clipped to between 0.65× and 1.6×.",
    })
  );
}

/* ---------------------------------------------------------------- Accuracy */

function accuracyCard() {
  const CASES = [
    { label: "Plated, separated, shot from above", band: "±15–20%", tone: "brand" },
    { label: "Bowl food, thick gravy, visible surface", band: "±20–30%", tone: "brand" },
    { label: "Stacked or layered — biryani, sandwiches", band: "±30–45%", tone: "warn" },
    { label: "Opaque container, food not visible", band: "unusable", tone: "warn" },
  ];

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "How wrong it can be" }),
      el("span", { class: "card__note", text: "Stated up front, not in a footnote" })
    ),
    el("p", {
      class: "small muted",
      text: "A single photograph has no absolute vertical scale, so depth is inferred rather than measured. These are the bands to expect for portion weight; calories inherit them, because the nutrition row itself is accurate to a few percent.",
    }),
    el(
      "div",
      { class: "stack stack--xs" },
      CASES.map((row) =>
        el(
          "div",
          { class: "settings-row" },
          el("div", { class: "settings-row__text small", text: row.label }),
          el(
            "div",
            { class: "settings-row__control" },
            el("span", { class: `chip chip--${row.tone}`, text: row.band })
          )
        )
      )
    ),
    el(
      "div",
      { class: "prose" },
      el("h3", { text: "What makes it better" }),
      el(
        "ul",
        {},
        el("li", { text: "Shoot from 30–45° above, not straight down and not level with the table — the model needs to see both footprint and relief." }),
        el("li", { text: "Get the whole plate in frame. The rim is the ruler; if it is cropped, the scale is guessed from the food instead." }),
        el("li", { text: "Set the real plate width. It is the only measurement you can give that the photo cannot." }),
        el("li", { text: "Correct a wrong dish name. The label picks the density and the nutrition row, so fixing it fixes the grams too." })
      )
    )
  );
}

/* ------------------------------------------------------------------ Engine */

function engineCard(health) {
  if (!health) return null;
  const full = health.engine === "full";

  return el(
    "section",
    { class: full ? "panel panel--info" : "panel panel--warn" },
    el(
      "div",
      { class: "row row--tight", style: { marginBottom: "0.35rem" } },
      icon(full ? "check" : "alert", { size: 16 }),
      el("strong", { text: full ? "This instance runs trained weights" : "This instance runs the built-in estimator" })
    ),
    el("p", {
      class: "small muted",
      text: full
        ? "Detection, depth and classification are all served by their trained models, so the accuracy bands above apply as written."
        : "No trained checkpoints are installed, so detection falls back to a plate-aware segmenter, depth to a distance-transform dome, and classification to a colour and texture signature. The geometry and the nutrition tables are unchanged; treat the dish names as a starting point and correct them where they are wrong.",
    }),
    el(
      "div",
      { class: "row row--tight" },
      Object.entries(health.models || {}).map(([key, info]) =>
        el("span", { class: "chip", text: `${key}: ${info.backend}` })
      )
    )
  );
}

/* ------------------------------------------------------------------ Limits */

function limitsCard(limits) {
  const rows = [
    ["Maximum upload", `${Math.round((limits.max_upload_bytes || 0) / 1048576)} MB`],
    ["Longest edge", `${num(limits.max_image_dimension || 0)} px, downscaled beyond that`],
    ["Items per plate", num(limits.max_items_per_plate || 0)],
    ["Flagged as unsure below", `${Math.round((limits.low_confidence_threshold || 0) * 100)}% confidence`],
    ["Left unidentified below", `${Math.round((limits.unrecognized_threshold || 0) * 100)}% confidence`],
    ["Accepted formats", (limits.allowed_mime || []).map((mime) => mime.replace("image/", "").toUpperCase()).join(", ")],
  ];

  if (!limits.max_upload_bytes) return null;

  return el(
    "details",
    { class: "card stack stack--sm" },
    el(
      "summary",
      { class: "row row--between", style: { cursor: "pointer" } },
      el("h2", { class: "card__title", text: "Operating limits" }),
      icon("chevronDown", { size: 17 })
    ),
    el(
      "div",
      { class: "kv" },
      rows.map(([label, value]) =>
        el("div", {}, el("div", { class: "kv__k", text: label }), el("div", { class: "kv__v", text: value }))
      )
    )
  );
}

/* ----------------------------------------------------------------- Sources */

function sourcesCard() {
  return el(
    "section",
    { class: "card stack stack--sm" },
    el("h2", { class: "card__title", text: "Where the nutrition comes from" }),
    el(
      "div",
      { class: "prose" },
      el("p", {
        html:
          "Per-100 g composition is taken from the <strong>Indian Food Composition Tables (IFCT 2017)</strong> for Indian dishes and " +
          "<strong>USDA FoodData Central</strong> for everything else. Densities are measured or sourced values in g/ml, not assumed to be 1.",
      }),
      el("p", {
        text: "Percentage daily values follow the FDA 2016 reference intakes for a 2,000 kcal diet. Sodium and added sugar are shown in amber past 100% rather than green, because for those two a high percentage is a warning, not an achievement.",
      }),
      el("p", {
        text: "Micronutrients are scaled linearly from the same row as the macros. That is a real simplification: cooking losses for vitamin C and folate are substantial and are not modelled. Read those two as an upper bound.",
      })
    ),
    el(
      "div",
      { class: "row row--tight" },
      el("a", { href: "/", class: "btn btn--primary" }, icon("camera", { size: 16 }), el("span", { text: "Analyse a meal" })),
      el("a", { href: "/docs", class: "btn btn--ghost", "data-native": "true" }, icon("book", { size: 16 }), el("span", { text: "API reference" }))
    )
  );
}
