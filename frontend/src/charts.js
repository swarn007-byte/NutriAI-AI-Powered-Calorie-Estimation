/* Hand-rolled SVG charts. Three shapes, no library.
 *
 * All of them are pure functions of their data and return a detached SVG node,
 * so they can be re-rendered on every update without a diffing layer. They use
 * `viewBox` with no fixed width/height, which is what makes them responsive:
 * the parent decides the size and the geometry follows.
 */

import { svg, el, kcal, num, pct, clamp } from "./dom.js";

const TAU = Math.PI * 2;

/* -------------------------------------------------------------------- Donut */

/** Macro donut with a calorie total in the middle.
 *
 *  Slices are drawn as stroked arcs on a shared circle rather than filled
 *  wedge paths: one `stroke-dasharray` per slice, offset by the running total.
 *  That keeps the joins clean at any radius and makes the draw-on animation a
 *  single dashoffset transition per slice.
 */
export function donut(slices, { total, label, size = 176, thickness = 16, animate = true } = {}) {
  const radius = (size - thickness) / 2;
  const circumference = TAU * radius;
  const sum = slices.reduce((acc, slice) => acc + Math.max(0, slice.value), 0);

  const ring = svg("g", { transform: `rotate(-90 ${size / 2} ${size / 2})` });

  ring.appendChild(
    svg("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: "var(--surface-3)",
      "stroke-width": thickness,
    })
  );

  let offset = 0;
  slices.forEach((slice, index) => {
    const value = Math.max(0, slice.value);
    if (value <= 0 || sum <= 0) return;
    const fraction = value / sum;
    // A 1.2° gap between slices reads as separation without implying a missing
    // value; below ~2% of the ring the gap would consume the slice, so skip it.
    const gap = fraction > 0.02 ? circumference * 0.0035 : 0;
    const length = Math.max(0, circumference * fraction - gap);

    const arc = svg("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: slice.color,
      "stroke-width": thickness,
      "stroke-linecap": fraction > 0.03 ? "round" : "butt",
      "stroke-dasharray": `${length} ${circumference - length}`,
      "stroke-dashoffset": -(circumference * offset),
    });

    if (animate) {
      arc.style.transformOrigin = "center";
      arc.animate(
        [
          { strokeDasharray: `0 ${circumference}` },
          { strokeDasharray: `${length} ${circumference - length}` },
        ],
        { duration: 620, delay: index * 90, easing: "cubic-bezier(.22,1,.36,1)", fill: "backwards" }
      );
    }

    ring.appendChild(arc);
    offset += fraction;
  });

  const centre = svg(
    "g",
    { class: "donut__centre" },
    svg("text", {
      x: size / 2,
      y: size / 2 - 2,
      "text-anchor": "middle",
      "dominant-baseline": "central",
      fill: "var(--text)",
      "font-size": size * 0.2,
      "font-weight": 720,
      "letter-spacing": "-0.03em",
      style: "font-variant-numeric:tabular-nums",
      text: total !== undefined ? kcal(total) : "",
    }),
    label
      ? svg("text", {
          x: size / 2,
          y: size / 2 + size * 0.145,
          "text-anchor": "middle",
          "dominant-baseline": "central",
          fill: "var(--text-faint)",
          "font-size": size * 0.082,
          "font-weight": 700,
          "letter-spacing": "0.09em",
          text: label,
        })
      : null
  );

  return svg(
    "svg",
    {
      viewBox: `0 0 ${size} ${size}`,
      width: size,
      height: size,
      role: "img",
      "aria-label": ariaForSlices(slices, total),
      style: "max-width:100%;height:auto",
    },
    ring,
    centre
  );
}

function ariaForSlices(slices, total) {
  const sum = slices.reduce((acc, slice) => acc + Math.max(0, slice.value), 0) || 1;
  const parts = slices
    .filter((slice) => slice.value > 0)
    .map((slice) => `${slice.label} ${pct((slice.value / sum) * 100)}`);
  return `${total !== undefined ? `${kcal(total)} kcal. ` : ""}${parts.join(", ")}`;
}

/* --------------------------------------------------------------- Goal gauge */

/** A 270° open gauge for "calories today vs goal". Distinct from the donut on
 *  purpose: it answers one question, and an open arc reads as a scale with a
 *  start and an end rather than a composition of parts. */
export function gauge(value, goal, { size = 168, thickness = 14, label = "of goal" } = {}) {
  const radius = (size - thickness) / 2;
  const sweep = 0.75; // 270°
  const circumference = TAU * radius;
  const track = circumference * sweep;
  const ratio = goal > 0 ? clamp(value / goal, 0, 1.35) : 0;
  const filled = Math.min(track, track * ratio);
  const over = value > goal && goal > 0;

  const arcs = svg(
    "g",
    { transform: `rotate(135 ${size / 2} ${size / 2})` },
    svg("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: "var(--surface-3)",
      "stroke-width": thickness,
      "stroke-linecap": "round",
      "stroke-dasharray": `${track} ${circumference}`,
    }),
    svg("circle", {
      cx: size / 2,
      cy: size / 2,
      r: radius,
      fill: "none",
      stroke: over ? "var(--warn)" : "var(--brand)",
      "stroke-width": thickness,
      "stroke-linecap": "round",
      "stroke-dasharray": `${filled} ${circumference}`,
      style: "transition:stroke-dasharray .6s cubic-bezier(.22,1,.36,1)",
    })
  );

  return svg(
    "svg",
    {
      viewBox: `0 0 ${size} ${size}`,
      width: size,
      height: size,
      role: "img",
      "aria-label": `${kcal(value)} of ${kcal(goal)} kcal, ${pct(goal ? (value / goal) * 100 : 0)}`,
      style: "max-width:100%;height:auto",
    },
    arcs,
    svg("text", {
      x: size / 2,
      y: size / 2 - size * 0.04,
      "text-anchor": "middle",
      "dominant-baseline": "central",
      fill: "var(--text)",
      "font-size": size * 0.215,
      "font-weight": 740,
      "letter-spacing": "-0.03em",
      style: "font-variant-numeric:tabular-nums",
      text: kcal(value),
    }),
    svg("text", {
      x: size / 2,
      y: size / 2 + size * 0.115,
      "text-anchor": "middle",
      "dominant-baseline": "central",
      fill: "var(--text-faint)",
      "font-size": size * 0.085,
      "font-weight": 640,
      style: "font-variant-numeric:tabular-nums",
      text: `${kcal(goal)} ${label}`,
    })
  );
}

/* ----------------------------------------------------------------- Sparkline */

/** Trend line with an area fill and an end-point marker. Width is nominal — the
 *  viewBox scales it — but the aspect ratio is fixed so the slope stays honest. */
export function sparkline(values, { width = 320, height = 68, color = "var(--brand)" } = {}) {
  const points = values.map((value) => Number(value) || 0);
  if (points.length < 2) {
    return svg("svg", { viewBox: `0 0 ${width} ${height}`, "aria-hidden": "true" });
  }

  const max = Math.max(...points, 1);
  const pad = 4;
  const stepX = (width - pad * 2) / (points.length - 1);
  const y = (value) => height - pad - (value / max) * (height - pad * 2);
  const coords = points.map((value, index) => [pad + index * stepX, y(value)]);

  const line = coords.map(([x, yy], i) => `${i === 0 ? "M" : "L"}${x.toFixed(1)} ${yy.toFixed(1)}`).join(" ");
  const area = `${line} L${(width - pad).toFixed(1)} ${height - pad} L${pad} ${height - pad} Z`;
  const gradientId = `spark-${Math.random().toString(36).slice(2, 9)}`;

  const path = svg("path", {
    d: line,
    fill: "none",
    stroke: color,
    "stroke-width": 2.2,
    "stroke-linecap": "round",
    "stroke-linejoin": "round",
    "vector-effect": "non-scaling-stroke",
  });

  // Draw-on: dash the whole length, then animate the offset to zero.
  const length = coords.reduce(
    (acc, [x, yy], i) =>
      i === 0 ? 0 : acc + Math.hypot(x - coords[i - 1][0], yy - coords[i - 1][1]),
    0
  );
  path.setAttribute("stroke-dasharray", String(length));
  path.setAttribute("stroke-dashoffset", String(length));
  path.style.animation = "draw 900ms cubic-bezier(.22,1,.36,1) forwards";

  const last = coords[coords.length - 1];

  return svg(
    "svg",
    {
      viewBox: `0 0 ${width} ${height}`,
      preserveAspectRatio: "none",
      role: "img",
      "aria-label": `Trend, latest ${num(points[points.length - 1])}`,
      style: "width:100%;height:auto;overflow:visible",
    },
    svg(
      "defs",
      {},
      svg(
        "linearGradient",
        { id: gradientId, x1: "0", y1: "0", x2: "0", y2: "1" },
        svg("stop", { offset: "0%", "stop-color": color, "stop-opacity": "0.3" }),
        svg("stop", { offset: "100%", "stop-color": color, "stop-opacity": "0" })
      )
    ),
    svg("path", { d: area, fill: `url(#${gradientId})`, stroke: "none" }),
    path,
    svg("circle", {
      cx: last[0],
      cy: last[1],
      r: 3.4,
      fill: color,
      stroke: "var(--bg)",
      "stroke-width": 2,
    })
  );
}

/* --------------------------------------------------------------- Trend bars */

/** Day-by-day calorie bars. HTML rather than SVG: each column carries a label
 *  and a tooltip, and letting flexbox size them means no width arithmetic and
 *  no resize listener. The goal line rides on a CSS custom property so its
 *  position is computed against the bar track alone (see .trend in pages.css). */
export function trendBars(days, { goal = 0, onSelect } = {}) {
  const peak = Math.max(goal, ...days.map((day) => day.calories || 0), 1);

  const columns = days.map((day) => {
    const height = `${clamp(((day.calories || 0) / peak) * 100, day.calories ? 3 : 1.5, 100)}%`;
    const over = goal > 0 && day.calories > goal;
    const column = el(
      "div",
      {
        class: "trend__col",
        role: "listitem",
        "data-tip": `${day.label}: ${kcal(day.calories)} kcal${day.meals ? ` · ${day.meals} meal${day.meals === 1 ? "" : "s"}` : " · no meals"}`,
        tabindex: onSelect ? "0" : null,
      },
      el(
        "div",
        { class: "trend__track" },
        el("div", {
          class: ["trend__bar", !day.calories && "trend__bar--empty", over && "trend__bar--over"],
          style: { height },
        })
      ),
      el("div", { class: "trend__day", text: day.short })
    );
    if (onSelect) {
      column.style.cursor = "pointer";
      column.addEventListener("click", () => onSelect(day));
      column.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onSelect(day);
        }
      });
    }
    return column;
  });

  return el(
    "div",
    { class: "trend", role: "list", style: goal > 0 ? { "--goal-ratio": String(goal / peak) } : {} },
    columns,
    goal > 0
      ? el(
          "div",
          { class: "trend__goal" },
          el(
            "span",
            {
              class: "trend__goal-label",
              // Named as a whole, so dropping the word on narrow screens (pages.css)
              // is a purely visual economy and costs nothing to a screen reader.
              role: "img",
              "aria-label": `Goal ${kcal(goal)} kcal`,
            },
            el("span", { text: kcal(goal) }),
            el("span", { class: "trend__goal-word", text: " goal" })
          )
        )
      : null
  );
}

/* ------------------------------------------------------------ Macro segments */

export const MACROS = [
  { key: "protein_g", label: "Protein", color: "var(--protein)", kcalPerGram: 4 },
  { key: "carbs_g", label: "Carbs", color: "var(--carbs)", kcalPerGram: 4 },
  { key: "fat_g", label: "Fat", color: "var(--fat)", kcalPerGram: 9 },
];

/** Macro slices measured in *calories*, not grams.
 *
 *  This matters: 20 g of fat and 20 g of carbs are equal by mass but 180 vs 80
 *  kcal. A donut sliced by grams would misrepresent where the energy in the
 *  meal actually came from, which is the only question it is there to answer.
 */
export const macroSlices = (source) =>
  MACROS.map((macro) => ({
    label: macro.label,
    color: macro.color,
    value: (Number(source?.[macro.key]) || 0) * macro.kcalPerGram,
    grams: Number(source?.[macro.key]) || 0,
  }));

/** The thin three-colour bar used on cards and item rows. */
export function macroBar(source) {
  const slices = macroSlices(source);
  const sum = slices.reduce((acc, slice) => acc + slice.value, 0);
  if (sum <= 0) return el("div", { class: "macrobar" });
  return el(
    "div",
    {
      class: "macrobar",
      role: "img",
      "aria-label": ariaForSlices(slices),
    },
    slices.map((slice) =>
      el("div", {
        class: "macrobar__seg",
        style: { flexGrow: String(slice.value), background: slice.color },
      })
    )
  );
}
