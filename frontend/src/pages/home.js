/* Home — the upload screen, and the only page that matters on first visit.
 *
 * Three ways in (file picker, drag-and-drop, paste) plus a camera capture on
 * mobile and four samples for anyone who arrived without a photo. All of them
 * converge on `choose(file)`, so validation and preview exist once.
 *
 * This page asks for one thing: the photo. The plate width and the note used to
 * live here too, and both were questions asked too early — the width scales
 * weights for a list of items the user had not seen yet, and there was nothing
 * to write a note about. They moved to the review step, which is where the user
 * finally has the found items in front of them.
 */

import { el, icon, humanize } from "../dom.js";
import { navigate } from "../router.js";
import { setPending, peekPending, release } from "../pending.js";
import { api } from "../api.js";
import { ensureCatalog, limit, setMeal, setSession, state } from "../store.js";
import { toastError } from "../toast.js";

const SAMPLES = [
  { file: "thali.jpg", label: "Full thali" },
  { file: "curry-bowl.jpg", label: "Curry bowl" },
  { file: "dosa.jpg", label: "Dosa plate" },
  { file: "breakfast.jpg", label: "Breakfast" },
];

const STEPS = [
  {
    title: "Find every item",
    text: "Segments the plate and separates each dish, so a thali is five foods rather than one average.",
    model: "YOLOv8-seg",
  },
  {
    title: "Read the depth",
    text: "A relative depth map turns the flat photo into a height field — the part a bounding box alone cannot give you.",
    model: "MiDaS v3",
  },
  {
    title: "Measure the portion",
    text: "Plate diameter sets the scale, then footprint × mean depth gives volume, and density gives grams.",
    model: "geometry",
  },
  {
    title: "Name the dish",
    text: "Classifies the crop against a food catalogue, because “curry” and “paneer butter masala” are 90 kcal apart per 100 g.",
    model: "EfficientNet-B3",
  },
  {
    title: "Cost the nutrition",
    text: "Looks the dish up per 100 g and scales to the estimated weight — calories, macros and 13 micronutrients.",
    model: "USDA + IFCT",
  },
];

/** A 40 KB sample rendered as "0.0 MB", which reads like a failed read rather
 *  than a small file. Switch unit below a megabyte. */
function fileSize(bytes) {
  if (bytes < 1048576) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

export default function home() {
  const page = el("div", { class: "page stack stack--lg" });

  const fileInput = el("input", {
    type: "file",
    accept: "image/jpeg,image/png,image/webp,image/heic,image/heif",
    "aria-hidden": "true",
    tabindex: "-1",
  });

  const cameraInput = el("input", {
    type: "file",
    accept: "image/*",
    // `environment` asks for the rear camera. Ignored on desktop, which falls
    // back to the ordinary picker — no branching needed.
    capture: "environment",
    "aria-hidden": "true",
    tabindex: "-1",
  });

  /* ------------------------------------------------------------- Validation */

  const maxBytes = limit("max_upload_bytes", 10 * 1024 * 1024);
  const allowedMime = limit("allowed_mime", [
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
    "image/heif",
  ]);

  /** Reject client-side what the server would reject anyway. The point is not
   *  security — the server still validates — it is that a 12 MB upload over a
   *  phone connection should not have to complete before being told no. */
  function reject(file) {
    if (!file) return "No file was selected.";
    const type = (file.type || "").toLowerCase();
    // An empty type happens for HEIC on some Android builds; fall back to the
    // extension rather than blocking a legitimate photo.
    if (type && !allowedMime.includes(type)) {
      return `${type.replace("image/", "").toUpperCase()} isn't supported. Use JPEG, PNG or WebP.`;
    }
    if (!type && !/\.(jpe?g|png|webp|heic|heif)$/i.test(file.name || "")) {
      return "That doesn't look like an image. Use JPEG, PNG or WebP.";
    }
    if (file.size === 0) return "That file is empty.";
    if (file.size > maxBytes) {
      return `That image is ${(file.size / 1048576).toFixed(1)} MB — the limit is ${Math.round(maxBytes / 1048576)} MB.`;
    }
    return null;
  }

  /* ---------------------------------------------------------------- Preview */

  const dropzone = el("div", {
    class: "dropzone",
    role: "button",
    tabindex: "0",
    "aria-label": "Choose a meal photo to analyse",
  });

  const previewHost = el("div", { class: "stack" });

  let chosen = null;
  let manualMode = false;
  let manualEntries = [{ label: "", weight: "" }];

  function choose(file) {
    const problem = reject(file);
    if (problem) {
      toastError(problem, { title: "Can't use that photo" });
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    setPending({ file, previewUrl });
    chosen = { file, previewUrl };
    paint();
  }

  function unchoose() {
    release();
    chosen = null;
    fileInput.value = "";
    cameraInput.value = "";
    paint();
  }

  /* ----------------------------------------------------------------- Submit */

  /* `phase: "scan"` picks the quick pass — detect and name, no depth, no
   * nutrition. It ends on the review screen, not the results. */
  function submit() {
    if (!chosen) return;
    setPending({ ...chosen, phase: "scan" });
    navigate("/analyzing");
  }

  /* ------------------------------------------------------------------ Paint */

  function paint() {
    if (manualMode) {
      previewHost.replaceChildren(manualForm());
      return;
    }
    if (!chosen) {
      previewHost.replaceChildren(dropzone, sampleStrip(choose));
      return;
    }

    const image = el("img", {
      src: chosen.previewUrl,
      alt: "The photo you selected",
      decoding: "async",
    });
    // Chrome and Firefox cannot decode HEIC even though the API accepts it, so
    // a broken-image icon is a real possibility on an Android iPhone-photo
    // hand-off. Say what happened instead of showing a torn page.
    image.addEventListener("error", () => {
      image.replaceWith(
        el(
          "div",
          { class: "empty", style: { aspectRatio: "4 / 3" } },
          icon("image", { size: 26 }),
          el("p", { class: "small muted", text: `${chosen.file.name} — this browser can't preview this format, but it will still be analysed.` })
        )
      );
    });

    previewHost.replaceChildren(
      el(
        "div",
        { class: "preview" },
        image,
        el(
          "button",
          {
            class: "btn btn--sm btn--icon preview__clear",
            "aria-label": "Remove this photo",
            onclick: unchoose,
          },
          icon("x", { size: 16 })
        )
      ),
      el(
        "div",
        { class: "row row--between" },
        el("span", {
          class: "small faint",
          text: `${chosen.file.name} · ${fileSize(chosen.file.size)}`,
        }),
        el(
          "button",
          { class: "btn btn--ghost btn--sm", onclick: unchoose },
          icon("refresh", { size: 15 }),
          el("span", { text: "Choose another" })
        )
      ),
      el(
        "button",
        { class: "btn btn--primary btn--lg btn--block", onclick: submit, "data-autofocus": "true" },
        icon("sparkles", { size: 18 }),
        el("span", { text: "See what's on the plate" })
      ),
      // The next screen is a list to check, not a number. Saying so here is the
      // difference between a user who reviews the list and one who is surprised
      // by it.
      el("p", {
        class: "xsmall faint center",
        text: "A quick pass first — you'll check the items and give the plate size before anything is weighed.",
      })
    );
  }

  /* -------------------------------------------------------------- Dropzone */

  dropzone.append(
    el("div", { class: "dropzone__icon" }, icon("upload")),
    el("div", { class: "dropzone__title", text: "Drop a meal photo here" }),
    el("div", {
      class: "dropzone__hint",
      text: "Shoot from slightly above, with the whole plate in frame. JPEG, PNG or WebP up to " +
        `${Math.round(maxBytes / 1048576)} MB.`,
    }),
    el(
      "div",
      { class: "dropzone__actions" },
      el(
        "button",
        {
          class: "btn btn--primary",
          onclick: (event) => {
            event.stopPropagation();
            fileInput.click();
          },
        },
        icon("image", { size: 17 }),
        el("span", { text: "Choose photo" })
      ),
      el(
        "button",
        {
          class: "btn btn--outline",
          onclick: (event) => {
            event.stopPropagation();
            cameraInput.click();
          },
        },
        icon("camera", { size: 17 }),
        el("span", { text: "Take photo" })
      )
      el(
        "button",
        {
          class: "btn btn--ghost",
          onclick: (event) => {
            event.stopPropagation();
            manualMode = true;
            ensureCatalog().then(paint).catch((error) => {
              toastError(error?.message || "The food list couldn't be loaded.", { title: "Manual entry unavailable" });
              paint();
            });
            paint();
          },
        },
        el("span", { text: "Enter meal manually" })
      )
    ),
    fileInput,
    cameraInput
  );

  fileInput.addEventListener("change", () => choose(fileInput.files?.[0]));
  cameraInput.addEventListener("change", () => choose(cameraInput.files?.[0]));

  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      fileInput.click();
    }
  });

  function manualForm() {
    const catalog = state.catalog?.items || state.catalog || [];
    const rows = manualEntries.map((entry, index) => {
      const select = el(
        "select",
        { class: "input", "aria-label": `Food ${index + 1}` },
        el("option", { value: "", text: "Choose a food" }),
        catalog.map((food) =>
          el("option", {
            value: food.label,
            text: food.display_name || humanize(food.label),
            selected: food.label === entry.label,
          })
        )
      );
      select.addEventListener("change", () => {
        entry.label = select.value;
      });
      const weight = el("input", {
        class: "input",
        type: "number",
        min: "1",
        max: "3000",
        step: "1",
        inputmode: "decimal",
        placeholder: "Weight in grams",
        value: entry.weight,
        "aria-label": `Weight for food ${index + 1} in grams`,
      });
      weight.addEventListener("input", () => {
        entry.weight = weight.value;
      });
      return el("div", { class: "row row--tight" }, el("div", { style: { flex: "1 1 12rem" } }, select), el("div", { style: { flex: "0 1 10rem" } }, weight));
    });

    const message = el("p", { class: "small muted", text: catalog.length ? "Choose each food and enter its weight. You can add as many foods as you need." : "Loading the food list…" });
    const submit = el("button", { class: "btn btn--primary btn--lg btn--block", disabled: !catalog.length }, icon("check", { size: 18 }), el("span", { text: "Save meal" }));
    submit.addEventListener("click", async () => {
      const items = manualEntries.map((entry) => ({ label: entry.label, weight_g: Number(entry.weight) }));
      if (items.some((item) => !item.label || !Number.isFinite(item.weight_g) || item.weight_g <= 0)) {
        toastError("Choose a food and enter a weight for every row.", { title: "Complete the meal" });
        return;
      }
      submit.disabled = true;
      try {
        const result = await api.manualMeal({ items });
        if (result.token) setSession({ token: result.token, user: result.user });
        setMeal(result);
        navigate("/results");
      } catch (error) {
        submit.disabled = false;
        toastError(error?.message || "The meal couldn't be saved.", { title: "Not saved" });
      }
    });

    return el(
      "section",
      { class: "card stack stack--sm" },
      el("div", { class: "row row--between" }, el("div", {}, el("span", { class: "eyebrow", text: "No photo needed" }), el("h2", { class: "card__title", text: "Enter your meal" })), el("button", { class: "btn btn--ghost btn--sm", onclick: () => { manualMode = false; paint(); }, text: "Use a photo" })),
      message,
      el("div", { class: "stack stack--xs" }, rows),
      el("button", { class: "btn btn--outline", onclick: () => { manualEntries.push({ label: "", weight: "" }); paint(); } }, icon("plus", { size: 16 }), el("span", { text: "Add another food" })),
      submit
    );
  }

  /* dragenter/dragleave fire for every child element the cursor crosses, so a
   * naive pair of handlers makes the highlight flicker. Counting depth fixes
   * it, and is why this is not two one-liners. */
  let depth = 0;
  dropzone.addEventListener("dragenter", (event) => {
    event.preventDefault();
    depth += 1;
    dropzone.classList.add("is-over");
  });
  dropzone.addEventListener("dragover", (event) => {
    event.preventDefault();
  });
  dropzone.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (depth === 0) dropzone.classList.remove("is-over");
  });
  dropzone.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    dropzone.classList.remove("is-over");
    const file = event.dataTransfer?.files?.[0];
    if (file) choose(file);
  });

  /* Paste. Screenshots and copied images are a real path in — and the listener
   * is on the document, so it works without focusing the dropzone first. */
  const onPaste = (event) => {
    if (chosen) return;
    const item = Array.from(event.clipboardData?.items || []).find((entry) =>
      entry.type.startsWith("image/")
    );
    if (!item) return;
    const file = item.getAsFile();
    if (file) choose(file);
  };
  document.addEventListener("paste", onPaste);
  // The router replaces the outlet's children wholesale, so there is no unmount
  // hook — drop the listener the first time a navigation leaves this page.
  document.addEventListener("route:changed", function off() {
    if (page.isConnected) return;
    document.removeEventListener("paste", onPaste);
    document.removeEventListener("route:changed", off);
  });

  /* -------------------------------------------------------------- Assemble */

  page.append(
    el(
      "section",
      { class: "hero" },
      el("h1", { class: "hero__title" }, "Know what's on the plate. ", el(
        "span",
        { class: "hero__accent", text: "Down to the gram." }
      )),
      el("p", {
        class: "hero__lede",
        text: "Photograph a meal and get every item found, weighed and costed — calories, macros and micronutrients, with the measurement shown rather than hidden behind a single number.",
      })
    ),
    previewHost,
    el(
      "section",
      { class: "section" },
      el(
        "div",
        { class: "section__head" },
        el("h2", { class: "section__title", text: "How the estimate is made" }),
        el("a", { class: "textlink", href: "/method", text: "Read the method →" })
      ),
      el(
        "div",
        { class: "steps stagger" },
        STEPS.map((step) =>
          el(
            "article",
            { class: "step" },
            el("div", { class: "step__title", text: step.title }),
            el("p", { class: "step__text", text: step.text }),
            el("code", { class: "step__model", text: step.model })
          )
        )
      )
    ),
    accuracyNote()
  );

  // Restore a photo chosen before a navigation away and back.
  const carried = peekPending();
  if (carried?.file) {
    chosen = { file: carried.file, previewUrl: carried.previewUrl };
  }
  paint();

  return page;
}

/* ------------------------------------------------------------------ Samples */

function sampleStrip(choose) {
  const strip = el("div", { class: "samples" });

  const load = async (entry, button) => {
    button.setAttribute("aria-busy", "true");
    try {
      const response = await fetch(`/samples/${entry.file}`);
      if (!response.ok) throw new Error(String(response.status));
      const blob = await response.blob();
      choose(new File([blob], entry.file, { type: blob.type || "image/jpeg" }));
    } catch {
      toastError("That sample photo couldn't be loaded.");
    } finally {
      button.removeAttribute("aria-busy");
    }
  };

  strip.append(
    ...SAMPLES.map((entry) => {
      const button = el(
        "button",
        { class: "sample", "aria-label": `Try the ${entry.label} sample` },
        el("img", {
          src: `/samples/${entry.file}`,
          alt: "",
          loading: "lazy",
          decoding: "async",
        }),
        el("span", { class: "sample__label", text: entry.label })
      );
      button.addEventListener("click", () => load(entry, button));
      return button;
    })
  );

  return el(
    "section",
    { class: "section" },
    el(
      "div",
      { class: "section__head" },
      el("span", { class: "eyebrow", text: "No photo to hand?" }),
      el("span", { class: "section__note", text: "Try one of these" })
    ),
    strip
  );
}

/* ---------------------------------------------------------- Honesty panel */

/** Stating the error bar up front is a product decision, not a disclaimer.
 *  A portion estimate from a single photo is ±20–25%, and a user who learns
 *  that from the first screen trusts the numbers they do get. */
function accuracyNote() {
  const engine = state.health?.engine;
  return el(
    "div",
    { class: "panel panel--info" },
    el(
      "div",
      { class: "row row--tight", style: { marginBottom: "0.35rem" } },
      icon("info", { size: 16 }),
      el("strong", { text: "What to expect" })
    ),
    el("p", {
      class: "small muted",
      text:
        "Portion weight from a single photo lands within about 20–25% of the truth for plated food, and worse for anything stacked or in an opaque container. Every item shows how its weight was reached, and you can correct any of them — a correction is remembered for that meal." +
        (engine && engine !== "full"
          ? " This instance is running the built-in estimator rather than trained weights, so treat the labels as a starting point."
          : ""),
    })
  );
}
