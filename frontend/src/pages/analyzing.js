/* Processing — the wait between one phase's request and its answer.
 *
 * The stepper is the honest kind of progress indicator: its stages are the real
 * pipeline stages, and each one is marked done only when the response comes back
 * with a measured timing for it. Until then the *current* stage is advanced on a
 * schedule derived from the typical cost of each stage, because the backend
 * returns one response, not a stream — so this is an estimate of where the work
 * is, and it says so by never claiming a stage is finished.
 *
 * This page serves both phases of the two-phase flow. The scan is short and
 * ends at the review screen; the deep pass is the expensive half and ends at
 * the results. Which one runs is decided by `pending.phase`, because the two
 * differ only in their stage list, their request and their destination.
 */

import { el, icon, num } from "../dom.js";
import { navigate } from "../router.js";
import { api, ApiError } from "../api.js";
import { takePending, setPending } from "../pending.js";
import { setDraft, setMeal, setSession } from "../store.js";
import { announce } from "../components/shell.js";
import { toastError } from "../toast.js";

/* Relative weights, not absolute times: detection dominates the scan, depth the
 * deep pass, and nutrition lookup is a dictionary hit. Used only to pace the
 * visual stepper. The keys are the backend's own `timings_ms` keys, which is what
 * lets a stage claim it is done only once there is a measured cost for it. */

/* The scan skips MiDaS entirely — depth is only useful once there is a plate
 * scale to turn it into volume — so it is roughly half the work and says so. */
export const SCAN_STAGES = [
  { key: "detection", label: "Finding the food", weight: 0.45 },
  { key: "classification", label: "Identifying dishes", weight: 0.4 },
  { key: "total", label: "Listing what it found", weight: 0.15 },
];

export const DEEP_STAGES = [
  { key: "depth", label: "Reading the depth", weight: 0.5 },
  { key: "volume", label: "Measuring portions", weight: 0.3 },
  { key: "nutrition", label: "Costing the nutrition", weight: 0.2 },
];

const NOMINAL_MS = 4200;
const SCAN_NOMINAL_MS = 2400;

const PHASES = {
  scan: {
    stages: SCAN_STAGES,
    nominalMs: SCAN_NOMINAL_MS,
    title: "Looking at your meal",
    blurb:
      "A quick pass to see what's on the plate. You'll get to check the list before anything is weighed or counted.",
    done: "Scan complete",
  },
  deep: {
    stages: DEEP_STAGES,
    nominalMs: NOMINAL_MS,
    title: "Measuring your meal",
    blurb:
      "Now the slow half: depth from the photo, portions from the plate you gave, then the nutrition.",
    done: "Analysis complete",
  },
};

export default function analyzing() {
  const carried = takePending();
  // Anything that is not the deep pass is a scan. There is no third case: the
  // one-shot endpoint still exists on the server, but no screen reaches it, and
  // defaulting an unknown phase to "deep" would 404 for want of a draft id.
  const deep = carried?.phase === "deep";
  const phase = deep ? PHASES.deep : PHASES.scan;

  // A deep pass needs a draft id; a scan needs the file itself. Either way, a
  // direct hit or a refresh has nothing to work on.
  const ready = deep ? Boolean(carried?.draftId) : Boolean(carried?.file);
  if (!ready) {
    navigate("/", { replace: true });
    return el("div", { class: "page" });
  }

  const stages = phase.stages;
  const page = el("div", { class: "page" });
  const stagelist = el("div", { class: "stagelist", role: "list" });
  const rows = stages.map((stage, index) =>
    el(
      "div",
      { class: "stage", role: "listitem" },
      el("span", { class: "stage__mark", text: String(index + 1) }),
      el("span", { class: "stage__label", text: stage.label }),
      el("span", { class: "stage__meta" })
    )
  );
  stagelist.append(...rows);

  let active = -1;

  function enter(index) {
    if (index <= active) return;
    active = index;
    rows.forEach((row, i) => {
      row.classList.toggle("is-active", i === index);
      row.classList.toggle("is-done", i < index);
      if (i < index) {
        const mark = row.querySelector(".stage__mark");
        if (!mark.querySelector("svg")) mark.replaceChildren(icon("check", { size: 13 }));
      }
    });
    announce(stages[index].label);
  }

  /* Advance on a timer weighted by each stage's typical share of the total, and
   * hold on the last stage rather than completing — a bar that reaches 100% and
   * then waits is worse than one that is visibly still working. */
  const timers = [];
  let elapsed = 0;
  stages.forEach((stage, index) => {
    timers.push(setTimeout(() => enter(index), elapsed));
    elapsed += stage.weight * phase.nominalMs;
  });

  const controller = new AbortController();
  const stopTimers = () => timers.forEach(clearTimeout);

  // Cancelling the deep pass should land back on the list the user just edited,
  // not at the upload screen — the draft still exists on the server, and their
  // edits are still in sessionStorage.
  const cancelHref = deep ? "/review" : "/";

  const cancel = el(
    "button",
    { class: "btn btn--ghost btn--sm" },
    icon("x", { size: 15 }),
    el("span", { text: "Cancel" })
  );
  cancel.addEventListener("click", () => {
    controller.abort();
    stopTimers();
    navigate(cancelHref, { replace: true });
  });

  const frame = el(
    "div",
    { class: "scanframe scanframe__corners" },
    carried.previewUrl ? el("img", { src: carried.previewUrl, alt: "", decoding: "async" }) : null,
    el("div", { class: "scanline" })
  );

  page.append(
    el(
      "div",
      { class: "processing" },
      el(
        "div",
        { class: "center stack stack--sm" },
        el("h1", { style: { fontSize: "var(--step-2)" }, text: phase.title }),
        el("p", {
          class: "muted small",
          style: { maxWidth: "44ch" },
          text: phase.blurb,
        })
      ),
      frame,
      stagelist,
      cancel
    )
  );

  /* ------------------------------------------------------------------ Request */

  const requested = deep
    ? api.analyzeDraft(
        carried.draftId,
        {
          plate_diameter_cm: carried.plateCm,
          notes: carried.notes || null,
          items: carried.edits || [],
        },
        { signal: controller.signal }
      )
    : api.scan({
        file: carried.file,
        notes: carried.notes,
        signal: controller.signal,
      });

  requested
    .then((result) => {
      stopTimers();
      // Mark every stage done and show the real per-stage timings, so the last
      // thing the user sees before the result is what it actually cost.
      const timings = result.timings_ms || {};
      rows.forEach((row, index) => {
        row.classList.remove("is-active");
        row.classList.add("is-done");
        row.querySelector(".stage__mark").replaceChildren(icon("check", { size: 13 }));
        const ms = timings[stages[index].key];
        if (ms !== undefined) row.querySelector(".stage__meta").textContent = `${num(ms)} ms`;
      });
      announce(phase.done);

      // Both endpoints mint a token, because an upload is allowed to be the
      // thing that creates the guest. Adopt it or the draft belongs to a user
      // this client cannot prove it is, and the deep pass would 404.
      if (result.token) setSession({ token: result.token, user: result.user });

      if (!deep) {
        setDraft(result);
        // The blob preview has done its job — the review page shows the stored
        // photo, which is the same pixels the masks were computed from.
        if (carried.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(carried.previewUrl);
        setTimeout(() => navigate("/review", { replace: true }), 380);
        return;
      }

      setMeal(result);
      // The draft is consumed server-side by a successful deep pass.
      setDraft(null);
      // The preview blob is pinned in memory until revoked, and the results page
      // uses the server's own image URL from here on.
      if (carried.previewUrl?.startsWith("blob:")) URL.revokeObjectURL(carried.previewUrl);
      // A short beat so the completed stepper is legible rather than a flash.
      setTimeout(() => navigate("/results", { replace: true }), 420);
    })
    .catch((error) => {
      stopTimers();
      if (error?.name === "AbortError") return;
      page.replaceChildren(failure(error, carried));
    });

  return page;
}

/* ------------------------------------------------------------------ Failure */

/** The error contract in one place. Each status means something different to
 *  the user and only one of them is worth retrying, so a single "try again"
 *  screen would be wrong for four of the five cases. */
function failure(error, carried) {
  const status = error instanceof ApiError ? error.status : 0;
  // A deep pass that failed still has a draft and a reviewed list behind it, so
  // its way out is back to that list — not back to the camera.
  const deep = carried?.phase === "deep";

  const COPY = {
    422: {
      title: deep ? "Nothing left to analyse" : "No food in that photo",
      body: deep
        ? error.message
        : "Nothing on the plate could be identified as food. A clearer shot from slightly above, with the whole plate in frame and the food filling most of it, usually fixes this.",
      icon: "image",
    },
    400: {
      title: "That image couldn't be used",
      body: error.message,
      icon: "alert",
    },
    503: {
      title: "The models aren't ready",
      body: "The analysis stack is still loading, or is unavailable on this instance. Your photo is still here — try again in a moment.",
      icon: "clock",
    },
    404: {
      title: "That scan has expired",
      body: "Scans are kept for a few hours and then swept. Take the photo again — it only takes one pass.",
      icon: "clock",
    },
    401: {
      title: "Session expired",
      body: "Sign in again, or continue as a guest, and re-run the analysis.",
      icon: "login",
    },
  };

  const copy = COPY[status] || {
    title: status === 0 ? "Can't reach the server" : "That didn't work",
    body: error?.message || "Something failed on the way through. Nothing was saved.",
    icon: "alert",
  };

  const retry = el(
    "button",
    { class: "btn btn--primary", "data-autofocus": "true" },
    icon("refresh", { size: 16 }),
    el("span", { text: "Try again" })
  );
  retry.addEventListener("click", () => {
    // The pending payload was consumed on entry; put it back before re-entering.
    if (carried?.file || carried?.draftId) {
      setPending(carried);
      navigate("/analyzing", { replace: true });
    } else {
      navigate("/", { replace: true });
    }
  });

  // 404 means the draft is gone, so retrying it cannot work — and neither can
  // going back to a list that no longer has a photo behind it.
  const retryable = status === 422 || status === 400 || status === 503 || status === 0;

  toastError(copy.body, { title: copy.title });

  return el(
    "div",
    { class: "empty" },
    el("div", { class: "empty__icon" }, icon(copy.icon, { size: 26 })),
    el("h1", { class: "empty__title", text: copy.title }),
    el("p", { class: "empty__text", text: copy.body }),
    el(
      "div",
      { class: "row", style: { justifyContent: "center" } },
      retryable && !(deep && status === 422) ? retry : null,
      deep && status !== 404
        ? el(
            "a",
            { href: "/review", class: "btn btn--outline", "data-autofocus": deep && status === 422 ? "true" : null },
            icon("edit", { size: 16 }),
            el("span", { text: "Back to the list" })
          )
        : el(
            "a",
            { href: "/", class: "btn btn--outline" },
            icon("camera", { size: 16 }),
            el("span", { text: "Use a different photo" })
          ),
      status === 422 && !deep
        ? el(
            "a",
            { href: "/method", class: "btn btn--ghost" },
            icon("book", { size: 16 }),
            el("span", { text: "Why this happens" })
          )
        : null
    )
  );
}
