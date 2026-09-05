/* Review — the step between "what is on the plate" and "what it cost".
 *
 * This page exists because of a specific failure. A photo of four samosas came
 * back with a lemon wedge costed as Fish Curry and a basket weave costed as
 * Curry or Gravy, 45 g each, ~142 kcal of food that was never on the plate. The
 * numbers were computed before the one person who could see the photo was asked
 * whether the list was right.
 *
 * So nothing here carries nutrition. The scan deliberately returns no calories:
 * a figure the user has not confirmed the *items* for is a figure they remember
 * whether or not it survives review. What this page collects is three things the
 * photo cannot supply on its own — which regions are actually food, how many
 * pieces there are of the countable ones, and how wide the plate is — and only
 * then does the expensive half run.
 *
 * Removal is undoable in place rather than confirmed with a dialog. Deleting a
 * phantom is the most common action on this screen, and it should cost one click
 * and no thought.
 */

import { el, icon, num, humanize, clamp, kcal, mount } from "../dom.js";
import { navigate } from "../router.js";
import { plateView, warningPanel } from "../components/meal.js";
import { setPending } from "../pending.js";
import { state, setDraft, restoreDraft, ensureCatalog, preference } from "../store.js";
import { announce } from "../components/shell.js";
import { openSheet, toastError } from "../toast.js";

/* Matches `ItemEdit.piece_count` (ge=1, le=60) on the server. A plate with more
 * than sixty of anything is not a plate. */
const MAX_PIECES = 60;

/* Matches `pipeline.MAX_PIECE_GUESS`. Anything above this stops being a count
 * and starts being a heap, so the guess refuses rather than inventing 30. */
const MAX_GUESS = 12;

export default function review() {
  const draft = state.draft || restoreDraft();
  if (!draft?.draft_id) return noDraft();

  /* The scan already ran at a provisional width — the saved preference, or the
   * instance default. Start there so the counts guessed from area match the
   * scale shown, and let the user correct it with the real plate in view. */
  let plateCm = Number(draft.plate_diameter_cm) || Number(preference("plate_diameter_cm", 26)) || 26;
  let notes = draft.notes || "";
  let handAdded = 0;

  const rows = (draft.items || []).map(scannedRow);

  // The catalogue is needed to rename a row and to add one. Fetch it now rather
  // than on the first click, so the sheet opens populated.
  ensureCatalog().catch(() => {
    /* The sheet says so when it happens. */
  });

  const page = el("div", { class: "page stack stack--lg review-page" });
  const plateHost = el("div", { class: "stack stack--sm" });
  const list = el("div", { class: "revlist" });
  const lowHost = el("div");

  let plate = null;

  /* ----------------------------------------------------------- Cross-linking */

  const highlightRow = (key) => {
    for (const node of list.children) {
      if (node.dataset?.row) node.classList.toggle("is-active", node.dataset.row === key);
    }
  };

  /* ------------------------------------------------------------- The rows */

  /** A countable row's note has to answer "where did 4 come from, and what does
   *  it mean in grams" without pretending the grams are measured. */
  function countNote(row) {
    const each = `${num(row.pieceWeight, 0)} g each`;
    const total = row.count ? ` · about ${num(row.count * row.pieceWeight, 0)} g` : "";
    return `${each}${total}`;
  }

  function counter(row, onChange) {
    const input = el("input", {
      class: "counter__input",
      type: "number",
      inputmode: "numeric",
      min: "1",
      max: String(MAX_PIECES),
      step: "1",
      value: String(row.count ?? 1),
      "aria-label": `Number of pieces of ${row.displayName}`,
    });

    const apply = (next) => {
      const value = clamp(Math.round(Number(next) || 1), 1, MAX_PIECES);
      input.value = String(value);
      onChange(value);
    };

    const button = (delta, label, glyph) =>
      el(
        "button",
        {
          class: "counter__btn",
          type: "button",
          "aria-label": `${label} ${row.displayName}`,
          onclick: () => apply(Number(input.value) + delta),
        },
        icon(glyph, { size: 14 })
      );

    // `change` rather than `input`: typing "12" fires input at "1" and would
    // clamp-and-rewrite the field out from under the second keystroke.
    input.addEventListener("change", () => apply(input.value));

    return el(
      "div",
      { class: "counter" },
      button(-1, "One fewer", "minus"),
      input,
      button(1, "One more", "plus"),
      el("span", { class: "counter__unit", text: row.count === 1 ? "piece" : "pieces" })
    );
  }

  function renderRow(row, position) {
    const node = el("div", {
      class: [
        "rev",
        row.lowConfidence && !row.renamed && "rev--low",
        row.removed && "rev--removed",
        row.addedByHand && "rev--added",
      ],
      dataset: { row: row.key },
    });

    /* The mark doubles as the key between list and photo: kept rows are numbered
     * in the same order as the boxes, and a removed row loses its number because
     * it has lost its box. */
    const mark = row.removed
      ? el("span", { class: "rev__mark rev__mark--out" }, icon("x", { size: 13 }))
      : row.addedByHand
        ? el("span", { class: "rev__mark rev__mark--new" }, icon("plus", { size: 13 }))
        : el("span", { class: "rev__mark", text: String(position + 1) });

    const note = el("div", { class: "rev__note", text: rowNote(row) });

    const body = el(
      "div",
      { class: "rev__body" },
      el(
        "div",
        { class: "rev__name" },
        el("span", { class: "truncate", text: row.displayName }),
        row.renamed
          ? el("span", { class: "chip chip--brand", "data-tip": "You named this", text: "Yours" })
          : null,
        row.lowConfidence && !row.renamed
          ? el("span", {
              class: "chip chip--warn",
              "data-tip": "The classifier was not sure. If this is not food, remove it.",
              text: `${Math.round((row.confidence || 0) * 100)}% sure`,
            })
          : null,
        row.guessed && row.count
          ? el("span", {
              class: "chip",
              "data-tip": "Counted from how much of the plate this region covers. Correct it if it is wrong.",
              text: "counted from area",
            })
          : null
      ),
      note
    );

    const controls = el("div", { class: "rev__side" });
    if (!row.removed) {
      if (row.pieceWeight !== null) {
        controls.appendChild(
          counter(row, (value) => {
            row.count = value;
            row.countTouched = true;
            row.guessed = false;
            note.textContent = rowNote(row);
            // The "counted from area" chip is now a lie; the name line owns it,
            // so re-render just this row's head rather than the whole list.
            const chip = [...body.querySelectorAll(".chip")].find(
              (element) => element.textContent === "counted from area"
            );
            chip?.remove();
            const unit = controls.querySelector(".counter__unit");
            if (unit) unit.textContent = value === 1 ? "piece" : "pieces";
          })
        );
      }
      controls.append(
        el(
          "button",
          {
            class: "btn btn--ghost btn--sm btn--icon",
            type: "button",
            "aria-label": `Rename ${row.displayName}`,
            "data-tip": "Wrong dish?",
            "data-rename": row.key,
            onclick: () => openRename(row),
          },
          icon("edit", { size: 15 })
        ),
        el(
          "button",
          {
            class: "btn btn--ghost btn--sm btn--icon rev__drop",
            type: "button",
            "aria-label": `Remove ${row.displayName}`,
            "data-tip": "Not on the plate",
            onclick: () => {
              row.removed = true;
              announce(`${row.displayName} removed`);
              render({ focusKey: row.key });
            },
          },
          icon("trash", { size: 15 })
        )
      );
    } else {
      controls.appendChild(
        el(
          "button",
          {
            class: "btn btn--outline btn--sm",
            type: "button",
            "data-rename": row.key,
            onclick: () => {
              row.removed = false;
              announce(`${row.displayName} back in`);
              render({ focusKey: row.key });
            },
          },
          icon("refresh", { size: 14 }),
          el("span", { text: "Undo" })
        )
      );
    }

    node.append(mark, body, controls);

    node.addEventListener("mouseenter", () => plate?.highlight(row.removed ? null : row.key));
    node.addEventListener("mouseleave", () => plate?.highlight(null));

    return node;
  }

  function rowNote(row) {
    if (row.removed) return "Removed — it won't be measured or costed.";
    if (row.pieceWeight !== null) return countNote(row);
    if (row.addedByHand) return "Added by hand — a standard serving, adjustable on the results.";
    return row.areaCm2 > 0
      ? `Portion measured from the photo · ${num(row.areaCm2, 0)} cm² of the plate`
      : "Portion measured from the photo";
  }

  /* --------------------------------------------------------------- Rename */

  /** Rename in place. The model's own runners-up come first because that is
   *  where the answer usually is — the classifier was often nearly right — and
   *  the whole catalogue is behind a search box for when it was not. */
  function openRename(row) {
    // Retried here as well as at mount: if the first fetch failed on a flaky
    // connection, the click that opens this sheet is a second chance, and the
    // catalogue is what decides whether the renamed food gets a count.
    ensureCatalog().catch(() => {});

    let chosen = null;
    const results = el("div", { class: "suggestions" });
    const alternatives = el("div", { class: "suggestions" });

    const save = el(
      "button",
      { class: "btn btn--primary", disabled: true },
      icon("check", { size: 16 }),
      el("span", { text: "Rename" })
    );

    const pick = (entry, node) => {
      chosen = entry;
      for (const button of [...alternatives.children, ...results.children]) {
        button.setAttribute("aria-pressed", String(button === node));
      }
      save.disabled = false;
    };

    if (row.alternatives.length) {
      alternatives.append(
        ...row.alternatives.map((alt) => {
          const button = el(
            "button",
            { class: "suggestion", "aria-pressed": "false" },
            el("span", { class: "suggestion__name", text: alt.display_name || humanize(alt.label) }),
            alt.confidence !== undefined
              ? el("span", { class: "suggestion__conf", text: `${Math.round(alt.confidence * 100)}%` })
              : null
          );
          button.addEventListener("click", () => pick({ label: alt.label, display_name: alt.display_name }, button));
          return button;
        })
      );
    }

    const search = el("input", {
      class: "input",
      type: "search",
      placeholder: "Search all foods…",
      "aria-label": "Search the food catalogue",
    });
    search.addEventListener("input", () => fillSearch(search, results, pick));

    const body = el(
      "div",
      { class: "stack stack--sm" },
      el(
        "div",
        { class: "row row--between" },
        el(
          "div",
          {},
          el("div", { class: "settings-row__title", text: row.displayName }),
          el("div", {
            class: "settings-row__note",
            text: row.detectedLabel && row.detectedLabel !== row.label
              ? `Found as ${humanize(row.detectedLabel)}`
              : rowNote(row),
          })
        ),
        row.lowConfidence && !row.renamed
          ? el("span", { class: "chip chip--warn", text: `${Math.round((row.confidence || 0) * 100)}% sure` })
          : null
      ),
      row.alternatives.length
        ? el("div", { class: "stack stack--sm" }, el("span", { class: "eyebrow", text: "Did you mean" }), alternatives)
        : null,
      el(
        "div",
        { class: "field" },
        el("span", { class: "field__label", text: "Or find the right food" }),
        search,
        results
      )
    );

    const cancel = el("button", { class: "btn btn--ghost", text: "Cancel" });
    const { close } = openSheet({ title: "What is this?", body, footer: [cancel, save] });
    cancel.addEventListener("click", () => close());

    save.addEventListener("click", () => {
      if (!chosen) return;
      rename(row, chosen);
      close();
      announce(`Renamed to ${row.displayName}`);
      render({ focusKey: row.key });
    });

    // Focus goes to the search box, not the first suggestion: picking a
    // suggestion is one keypress away either way, and typing is the fallback
    // people reach for when none of them is right.
    setTimeout(() => search.focus(), 60);
  }

  /* ------------------------------------------------------------- Add an item */

  function openAdd() {
    ensureCatalog().catch(() => {});

    let chosen = null;
    const results = el("div", { class: "suggestions" });

    const save = el(
      "button",
      { class: "btn btn--primary", disabled: true },
      icon("plus", { size: 16 }),
      el("span", { text: "Add it" })
    );

    const pick = (entry, node) => {
      chosen = entry;
      for (const button of results.children) button.setAttribute("aria-pressed", String(button === node));
      save.disabled = false;
    };

    const search = el("input", {
      class: "input",
      type: "search",
      placeholder: "Rice, dal, roti…",
      "aria-label": "Search the food catalogue",
    });
    search.addEventListener("input", () => fillSearch(search, results, pick));

    const body = el(
      "div",
      { class: "stack stack--sm" },
      el("p", {
        class: "small muted",
        text: "Something the scan missed — behind another dish, or off the edge of the frame. It has no region in the photo, so it gets a standard serving for its kind, or a piece count if you can give one.",
      }),
      el("div", { class: "field" }, search, results)
    );

    const cancel = el("button", { class: "btn btn--ghost", text: "Cancel" });
    const { close } = openSheet({ title: "Add an item", body, footer: [cancel, save] });
    cancel.addEventListener("click", () => close());

    save.addEventListener("click", () => {
      if (!chosen) return;
      handAdded += 1;
      const row = {
        index: -1,
        key: `add${handAdded}`,
        label: chosen.label,
        displayName: chosen.display_name || humanize(chosen.label),
        detectedLabel: null,
        category: chosen.category || "unknown",
        bbox: null,
        confidence: 1,
        lowConfidence: false,
        alternatives: [],
        areaCm2: 0,
        pieceWeight: chosen.piece_weight_g ?? null,
        footprint: chosen.piece_footprint_cm2 ?? null,
        count: chosen.piece_weight_g ? 1 : null,
        guessed: false,
        renamed: false,
        countTouched: false,
        removed: false,
        addedByHand: true,
      };
      rows.push(row);
      close();
      announce(`${row.displayName} added`);
      render({ focusKey: row.key });
    });

    setTimeout(() => search.focus(), 60);
  }

  /* ------------------------------------------------------------ Plate width */

  /* Moved here from the upload screen. Asking for the plate before the user has
   * seen what was found puts the one number that scales every weight in front of
   * someone who does not yet know whether the list is worth scaling. */
  function plateCard() {
    const output = el("b", { text: `${num(plateCm, 1)} cm` });
    const slider = el("input", {
      type: "range",
      class: "range",
      min: "12",
      max: "45",
      step: "0.5",
      value: String(plateCm),
      "aria-label": "Plate or bowl diameter in centimetres",
      oninput: (event) => {
        plateCm = Number(event.target.value);
        output.textContent = `${num(plateCm, 1)} cm`;
      },
    });

    return el(
      "div",
      { class: "card stack stack--sm" },
      el(
        "div",
        { class: "row row--between" },
        el(
          "div",
          {},
          el("div", { class: "settings-row__title", text: "Plate or bowl width" }),
          el("div", {
            class: "settings-row__note",
            text: "This sets the scale. A wrong width scales every measured weight with it.",
          })
        ),
        el("span", { class: "chip chip--brand" }, output)
      ),
      slider,
      el(
        "div",
        { class: "row row--between xsmall faint" },
        el("span", { text: "12 cm · small bowl" }),
        el("span", { text: "45 cm · sharing platter" })
      ),
      el("span", {
        class: "field__hint",
        text: "Piece counts are unaffected — the width only rescales the portions measured from the photo.",
      }),
      el(
        "label",
        { class: "field" },
        el("span", { class: "field__label", text: "Note (optional)" }),
        el("input", {
          class: "input",
          type: "text",
          maxlength: "280",
          placeholder: "Homemade, less oil…",
          value: notes,
          oninput: (event) => {
            notes = event.target.value;
          },
        })
      )
    );
  }

  /* ----------------------------------------------------------------- Submit */

  const cta = el(
    "button",
    { class: "btn btn--primary btn--lg btn--block", "data-autofocus": "true" },
    icon("sparkles", { size: 18 }),
    el("span", { text: "Measure and cost this" })
  );

  cta.addEventListener("click", () => {
    const kept = rows.filter((row) => !row.removed);
    if (!kept.length) {
      toastError("Add an item, or undo a removal, before measuring.", { title: "Nothing left to analyse" });
      return;
    }
    setPending({
      phase: "deep",
      draftId: draft.draft_id,
      plateCm,
      notes: notes.trim(),
      // The stored photo, so the progress screen shows the same pixels the masks
      // were computed from rather than nothing at all.
      previewUrl: draft.image_url,
      edits: rows.filter((row) => !(row.addedByHand && row.removed)).map(editFor),
    });
    navigate("/analyzing");
  });

  const startOver = el(
    "button",
    { class: "btn btn--ghost" },
    icon("camera", { size: 16 }),
    el("span", { text: "Different photo" })
  );
  startOver.addEventListener("click", () => {
    setDraft(null);
    navigate("/", { replace: true });
  });

  /* ----------------------------------------------------------------- Render */

  function render({ focusKey } = {}) {
    const kept = rows.filter((row) => !row.removed);

    plate = plateView(overlay(draft, kept), {
      onHover: (key) => highlightRow(key),
      onSelect: (key) => {
        const node = list.querySelector(`[data-row="${key}"] [data-rename]`);
        node?.focus();
        node?.scrollIntoView({ block: "center", behavior: "smooth" });
      },
    });
    plateHost.replaceChildren(plate);

    let position = 0;
    list.replaceChildren(
      ...rows.map((row) => renderRow(row, row.removed ? -1 : position++))
    );

    const unsure = kept.filter((row) => row.lowConfidence && !row.renamed).length;
    // mount(), not replaceChildren(): the panel is absent once every unsure row
    // has been renamed or removed, and replaceChildren(null) writes "null".
    mount(
      lowHost,
      unsure
        ? el(
            "div",
            { class: "panel panel--warn" },
            el(
              "div",
              { class: "row row--tight", style: { marginBottom: "0.3rem" } },
              icon("alert", { size: 16 }),
              el("strong", {
                text: unsure === 1 ? "One item isn't a confident match" : `${unsure} items aren't confident matches`,
              })
            ),
            el("p", {
              class: "small muted",
              text: "A shadow, a garnish or the plate rim can read as food. Anything on this list that isn't on the plate should go — it would otherwise be weighed and costed like the rest.",
            })
          )
        : null
    );

    cta.querySelector("span").textContent = kept.length
      ? `Measure and cost ${kept.length} item${kept.length === 1 ? "" : "s"}`
      : "Nothing to measure";
    cta.disabled = kept.length === 0;

    if (focusKey) list.querySelector(`[data-row="${focusKey}"] [data-rename]`)?.focus();
  }

  /* --------------------------------------------------------------- Assemble */

  // mount(), not append(): `warningPanel` returns null when the scan had nothing
  // to warn about, and native append() would render that as the text "null".
  mount(
    page,
    el(
      "div",
      { class: "review__head stack stack--xs" },
      el("span", { class: "eyebrow", text: "Step 2 of 3 · nothing costed yet" }),
      el("h1", { class: "review__title", text: "Is this what's on the plate?" }),
      el("p", {
        class: "muted",
        style: { maxWidth: "62ch" },
        text: "The scan found these regions and named them. Fix the names, say how many of anything countable, and drop whatever isn't food — then it gets weighed and costed.",
      })
    ),
    warningPanel(draft.warnings),
    lowHost,
    el(
      "div",
      { class: "split review__layout" },
      el("div", { class: "split__sticky stack stack--sm review__visual" }, plateHost),
      el(
        "div",
        { class: "stack stack--sm review__list" },
        el(
          "section",
          { class: "section" },
          el(
            "div",
            { class: "section__head" },
            el("h2", { class: "section__title", text: "What the scan found" }),
            el("span", { class: "section__note", text: "Hover a row to find it on the photo" })
          ),
          list
        ),
        el(
          "button",
          { class: "btn btn--outline btn--block", onclick: openAdd },
          icon("plus", { size: 16 }),
          el("span", { text: "Add something it missed" })
        ),
        plateCard(),
        el("div", { class: "row review__actions" }, startOver, el("span", { class: "grow" })),
        cta
      )
    )
  );

  render();
  return page;
}

/* -------------------------------------------------------------------- Rows */

/** One editable row. `index` is the scan's own index — the server matches edits
 *  by it, not by list position, so removing a row can't shift what a later edit
 *  refers to. `-1` is an item the user typed in. */
function scannedRow(item) {
  return {
    index: item.index,
    key: `scan${item.index}`,
    label: item.label,
    displayName: item.display_name,
    detectedLabel: item.detected_label,
    category: item.category,
    bbox: item.bbox || null,
    confidence: Number(item.confidence) || 0,
    lowConfidence: Boolean(item.low_confidence),
    alternatives: Array.isArray(item.alternatives) ? item.alternatives : [],
    areaCm2: Number(item.area_cm2) || 0,
    pieceWeight: item.piece_weight_g ?? null,
    footprint: null,
    count: item.piece_count ?? null,
    guessed: Boolean(item.piece_count_estimated),
    renamed: false,
    countTouched: false,
    removed: false,
    addedByHand: false,
  };
}

/** Apply a new name, and re-derive everything that hangs off the label.
 *
 *  A rename can change whether the food is countable at all — Curry or Gravy to
 *  Samosa turns a measured portion into four pieces — so the count is re-guessed
 *  from this region's own area rather than carried over from a different dish.
 *  This mirrors `_apply_edits` in main.py deliberately: the count shown here has
 *  to be the count the deep pass uses, or the review means nothing.
 */
function rename(row, chosen) {
  const entry = catalogEntry(chosen.label) || {};
  row.label = chosen.label;
  row.displayName = chosen.display_name || entry.display_name || humanize(chosen.label);
  row.category = entry.category || row.category;
  row.renamed = true;
  row.lowConfidence = false;
  row.confidence = 1;
  row.pieceWeight = entry.piece_weight_g ?? null;
  row.footprint = entry.piece_footprint_cm2 ?? null;
  row.countTouched = false;
  row.count = guessCount(row);
  row.guessed = row.count !== null;
}

/** `pipeline.guess_piece_count`, client-side, so the number on this screen is
 *  the number the server will reach for the same region. Both round a float, so
 *  the two can disagree at exactly x.5 — Python rounds to even, JS rounds up.
 *  Off by one piece in a case that needs three decimal places to reach; the
 *  count is editable, which is the actual answer to it. */
function guessCount(row) {
  if (row.pieceWeight === null || !row.footprint || row.areaCm2 <= 0) return null;
  return clamp(Math.round(row.areaCm2 / row.footprint), 1, MAX_GUESS);
}

const catalogEntry = (label) => (state.catalog || []).find((entry) => entry.label === label) || null;

/** The edit list. Only what the server has to act on:
 *
 *  - a label when the user renamed the row, or typed the item in
 *  - a count only when they actually set one. An untouched guess is left out on
 *    purpose: the server keeps its own guess *and* keeps flagging it as a guess,
 *    which is what the results page reads to say the weight is estimated. Send
 *    it back and it would silently become a measurement.
 */
function editFor(row) {
  const countable = row.pieceWeight !== null;
  return {
    index: row.index,
    label: row.addedByHand || row.renamed ? row.label : null,
    piece_count: countable && (row.countTouched || row.addedByHand) ? row.count : null,
    removed: row.removed,
  };
}

/** A meal-shaped object for `plateView`, which is the results page's overlay and
 *  is reused verbatim: same fractional bboxes, same viewBox trick, same
 *  hover/select contract. Rows without a bbox — the hand-added ones — are
 *  skipped by it, which is correct: they have no region in the photo. */
function overlay(draft, kept) {
  return {
    image_url: draft.image_url,
    image_width: draft.image_width,
    image_height: draft.image_height,
    items: kept.map((row) => ({
      id: row.key,
      display_name: row.displayName,
      bbox: row.bbox,
      low_confidence: row.lowConfidence,
      user_corrected: row.renamed,
    })),
  };
}

/* --------------------------------------------------------- Catalogue search */

/** Shared by the rename and add sheets: the same filter, the same eight results.
 *  Matching the label as well as the display name means "gulab_jamun" typed with
 *  a space still finds it. */
function fillSearch(search, host, pick) {
  const query = search.value.trim().toLowerCase();
  const catalog = state.catalog;

  if (!catalog) {
    host.replaceChildren(el("p", { class: "small muted", text: "The food list is still loading." }));
    return;
  }
  if (!query) {
    host.replaceChildren();
    return;
  }

  const entries = catalog.filter((entry) => {
    const name = String(entry.display_name || "").toLowerCase();
    return name.includes(query) || String(entry.label || "").includes(query.replace(/\s+/g, "_"));
  });

  if (!entries.length) {
    host.replaceChildren(el("p", { class: "small muted", text: `Nothing matching “${search.value.trim()}”.` }));
    return;
  }

  host.replaceChildren(
    ...entries.slice(0, 8).map((entry) => {
      const button = el(
        "button",
        { class: "suggestion", "aria-pressed": "false" },
        el(
          "span",
          { class: "suggestion__name" },
          el("span", { text: entry.display_name || humanize(entry.label) }),
          el("span", {
            class: "xsmall faint",
            style: { display: "block" },
            text: entry.piece_weight_g
              ? `${humanize(entry.category || "")} · counted, ${num(entry.piece_weight_g, 0)} g a piece`
              : humanize(entry.category || ""),
          })
        ),
        el("span", { class: "suggestion__conf", text: `${kcal(entry.kcal_per_100g)} /100g` })
      );
      button.addEventListener("click", () => pick(entry, button));
      return button;
    })
  );
}

/* ------------------------------------------------------------- No draft */

/** Landing here without a scan: a bookmark, a refresh hours later, or Back from
 *  the results after the draft was consumed. The last of those is the common one
 *  and deserves the meal it turned into rather than a redirect to the camera. */
function noDraft() {
  const meal = state.meal;
  return el(
    "div",
    { class: "page" },
    el(
      "div",
      { class: "empty" },
      el("div", { class: "empty__icon" }, icon(meal ? "check" : "camera", { size: 26 })),
      el("h1", { class: "empty__title", text: meal ? "That scan is already done" : "No scan open" }),
      el("p", {
        class: "empty__text",
        text: meal
          ? "A reviewed scan becomes a meal, and the scan itself is finished with. The meal it produced is still here."
          : "This is the step where you check what a scan found. Take or choose a photo and it will land here.",
      }),
      el(
        "div",
        { class: "row", style: { justifyContent: "center" } },
        meal
          ? el(
              "a",
              { href: "/results", class: "btn btn--primary" },
              icon("chart", { size: 16 }),
              el("span", { text: "See the results" })
            )
          : null,
        el(
          "a",
          { href: "/", class: meal ? "btn btn--outline" : "btn btn--primary" },
          icon("camera", { size: 16 }),
          el("span", { text: meal ? "Analyse another" : "Analyse a meal" })
        )
      )
    )
  );
}
