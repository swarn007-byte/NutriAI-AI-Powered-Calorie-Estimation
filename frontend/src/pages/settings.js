/* Settings — goals, scale, appearance, session.
 *
 * Everything saves on change rather than behind a Save button. There is no
 * multi-field transaction here: each preference is independent and the endpoint
 * is a PATCH of only the keys sent (main.py:301), so a Save button would add a
 * step and a "did it stick?" question for nothing. Writes are debounced and the
 * result is stated in a status line, which is what a Save button was really for.
 */

import { el, mount, icon, num, kcal, grams, clamp, debounce } from "../dom.js";
import { navigate } from "../router.js";
import { api } from "../api.js";
import {
  state,
  set,
  setTheme,
  setSession,
  clearSession,
  isGuest,
  displayName,
  preference,
} from "../store.js";
import { toastOk, toastError, confirmAction } from "../toast.js";

/* Bounds mirror schemas.Preferences exactly. Clamping here rather than letting
 * a 422 come back means the slider can never be in a state the server rejects. */
const BOUNDS = {
  calorie_goal: [800, 6000],
  protein_goal_g: [20, 400],
  carbs_goal_g: [20, 800],
  fat_goal_g: [10, 300],
  plate_diameter_cm: [12, 45],
};

/** The split the backend applies when a macro goal is unset (main.py:653). */
const DERIVED = {
  protein_goal_g: (kcalGoal) => Math.round((kcalGoal * 0.25) / 4),
  carbs_goal_g: (kcalGoal) => Math.round((kcalGoal * 0.5) / 4),
  fat_goal_g: (kcalGoal) => Math.round((kcalGoal * 0.25) / 9),
};

export default function settings() {
  const page = el("div", { class: "page stack stack--lg" });

  const status = el("span", { class: "xsmall faint", role: "status", "aria-live": "polite" });

  /* One writer for every control. The queue collapses rapid edits into a single
   * PATCH, and the response is the whole user object, so nothing is guessed. */
  let queued = {};
  const flush = debounce(async () => {
    const payload = queued;
    queued = {};
    if (!Object.keys(payload).length) return;
    status.textContent = "Saving…";
    try {
      const user = await api.savePreferences(payload);
      set({ user });
      status.textContent = "Saved";
      setTimeout(() => {
        if (status.textContent === "Saved") status.textContent = "";
      }, 2200);
    } catch (error) {
      status.textContent = "";
      toastError(error?.message || "That preference couldn't be saved.", { title: "Not saved" });
    }
  }, 600);

  const save = (patch) => {
    queued = { ...queued, ...patch };
    flush();
  };

  mount(
    page,
    el(
      "div",
      { class: "row row--between" },
      el(
        "div",
        {},
        el("span", { class: "eyebrow", text: "Preferences" }),
        el("h1", { style: { fontSize: "var(--step-3)" }, text: "Settings" })
      ),
      status
    ),
    accountCard(),
    goalsCard(save),
    scaleCard(save),
    appearanceCard(),
    instanceCard(),
    sessionCard()
  );

  return page;
}

/* ------------------------------------------------------------------ Account */

function accountCard() {
  const guest = isGuest();

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Account" }),
      guest ? el("span", { class: "chip chip--warn", text: "Guest" }) : el("span", { class: "chip chip--brand", text: "Signed in" })
    ),
    el(
      "div",
      { class: "settings-row" },
      el(
        "div",
        { class: "settings-row__text" },
        el("div", { class: "settings-row__title", text: displayName() }),
        el("div", {
          class: "settings-row__note",
          text: guest
            ? "A guest session lives in this browser only. Clearing site data ends it and the meals become unreachable."
            : state.user?.email || "",
        })
      ),
      el(
        "div",
        { class: "settings-row__control" },
        guest
          ? el(
              "a",
              { href: "/auth?mode=register&next=/settings", class: "btn btn--primary btn--sm" },
              icon("sparkles", { size: 15 }),
              el("span", { text: "Create an account" })
            )
          : el("span", { class: "small faint", text: `Member since ${memberSince()}` })
      )
    )
  );
}

const memberSince = () => {
  const created = state.user?.created_at;
  const date = created ? new Date(created) : null;
  return date && !Number.isNaN(date.getTime())
    ? date.toLocaleDateString(undefined, { month: "short", year: "numeric" })
    : "—";
};

/* -------------------------------------------------------------------- Goals */

function goalsCard(save) {
  let calorieGoal = clampTo("calorie_goal", Number(preference("calorie_goal", 2000)) || 2000);

  const macroRows = new Map();

  const calorieInput = numberInput({
    value: calorieGoal,
    step: 50,
    bounds: BOUNDS.calorie_goal,
    label: "Daily calorie goal in kcal",
    onCommit: (value) => {
      calorieGoal = value;
      save({ calorie_goal: value });
      // Any macro still on its derived default must follow the new calorie
      // figure, or the placeholder and the real target silently disagree.
      for (const [key, row] of macroRows) row.retrack(value);
    },
  });

  const rows = Object.keys(DERIVED).map((key) => {
    const row = macroRow(key, calorieGoal, save);
    macroRows.set(key, row);
    return row.node;
  });

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Daily goals" }),
      el("span", { class: "card__note", text: "Used by the Today gauge" })
    ),
    el(
      "div",
      { class: "settings-row" },
      el(
        "div",
        { class: "settings-row__text" },
        el("div", { class: "settings-row__title", text: "Calories" }),
        el("div", { class: "settings-row__note", text: `Between ${kcal(BOUNDS.calorie_goal[0])} and ${kcal(BOUNDS.calorie_goal[1])} kcal.` })
      ),
      el(
        "div",
        { class: "settings-row__control row row--tight" },
        el("div", { class: "goal-input" }, calorieInput),
        el("span", { class: "small faint", text: "kcal" })
      )
    ),
    ...rows,
    el("p", {
      class: "xsmall faint",
      text: "Leave a macro blank to track the default split — 25% protein, 50% carbohydrate, 25% fat by energy.",
    })
  );
}

/** One macro goal. Blank means "follow the calorie goal", which is the backend's
 *  own behaviour, so the placeholder shows the figure that would be used. */
function macroRow(key, calorieGoal, save) {
  let current = preference(key, null);
  const label = key.replace("_goal_g", "").replace(/^\w/, (c) => c.toUpperCase());

  const input = el("input", {
    class: "input",
    type: "number",
    min: String(BOUNDS[key][0]),
    max: String(BOUNDS[key][1]),
    step: "1",
    inputmode: "numeric",
    "aria-label": `${label} goal in grams`,
    value: current === null ? "" : String(current),
    placeholder: String(DERIVED[key](calorieGoal)),
  });

  input.addEventListener("change", () => {
    const raw = input.value.trim();
    if (!raw) {
      // The PATCH endpoint ignores nulls, so a cleared field cannot un-set a
      // stored goal. Say so rather than pretending the blank took effect.
      if (current !== null) {
        input.value = String(current);
        toastError("A goal that has been set can't be cleared — set it to the value you want instead.");
      }
      return;
    }
    const parsed = Math.round(Number(raw));
    if (!Number.isFinite(parsed)) {
      input.value = current === null ? "" : String(current);
      return;
    }
    const value = clampTo(key, parsed);
    input.value = String(value);
    if (value === current) return;
    current = value;
    save({ [key]: value });
  });

  return {
    node: el(
      "div",
      { class: "settings-row" },
      el(
        "div",
        { class: "settings-row__text" },
        el("div", { class: "settings-row__title", text: label }),
        el("div", {
          class: "settings-row__note",
          text: `${grams(BOUNDS[key][0])}–${grams(BOUNDS[key][1])} g`,
        })
      ),
      el(
        "div",
        { class: "settings-row__control row row--tight" },
        el("div", { class: "goal-input" }, input),
        el("span", { class: "small faint", text: "g" })
      )
    ),
    retrack(nextCalorieGoal) {
      input.placeholder = String(DERIVED[key](nextCalorieGoal));
    },
  };
}

function numberInput({ value, step, bounds, label, onCommit }) {
  const input = el("input", {
    class: "input",
    type: "number",
    inputmode: "numeric",
    min: String(bounds[0]),
    max: String(bounds[1]),
    step: String(step),
    value: String(value),
    "aria-label": label,
  });

  const commit = () => {
    const parsed = Math.round(Number(input.value));
    if (!Number.isFinite(parsed)) {
      input.value = String(value);
      return;
    }
    const next = Math.min(bounds[1], Math.max(bounds[0], parsed));
    input.value = String(next);
    if (next === value) return;
    value = next;
    onCommit(next);
  };

  input.addEventListener("change", commit);
  return input;
}

const clampTo = (key, value) => clamp(value, BOUNDS[key][0], BOUNDS[key][1]);

/* -------------------------------------------------------------------- Scale */

/** The plate diameter is the pixel→centimetre factor for every weight the app
 *  produces, so it earns a full card rather than a row. */
function scaleCard(save) {
  const initial = clampTo("plate_diameter_cm", Number(preference("plate_diameter_cm", 26)) || 26);
  const output = el("b", { text: `${num(initial, 1)} cm` });

  const slider = el("input", {
    type: "range",
    class: "range",
    min: String(BOUNDS.plate_diameter_cm[0]),
    max: String(BOUNDS.plate_diameter_cm[1]),
    step: "0.5",
    value: String(initial),
    "aria-label": "Default plate or bowl diameter in centimetres",
  });

  slider.addEventListener("input", () => {
    output.textContent = `${num(Number(slider.value), 1)} cm`;
  });
  slider.addEventListener("change", () => {
    save({ plate_diameter_cm: Number(slider.value) });
  });

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "Default plate width" }),
      el("span", { class: "chip chip--brand" }, output)
    ),
    el("p", {
      class: "small muted",
      text: "Pre-fills the slider on the upload screen. It sets the scale for every measurement — volume scales with the cube of this number, so a 10% error in it is a 33% error in every weight.",
    }),
    slider,
    el(
      "div",
      { class: "row row--between xsmall faint" },
      el("span", { text: "12 cm · small bowl" }),
      el("span", { text: "26 cm · dinner plate" }),
      el("span", { text: "45 cm · platter" })
    )
  );
}

/* --------------------------------------------------------------- Appearance */

function appearanceCard() {
  const OPTIONS = [
    { value: "dark", label: "Dark", iconName: "moon" },
    { value: "light", label: "Light", iconName: "sun" },
    { value: "system", label: "System", iconName: "settings" },
  ];

  const group = el("div", { class: "segmented", role: "group", "aria-label": "Colour theme" });

  const buttons = OPTIONS.map((option) => {
    const button = el(
      "button",
      {
        class: "segmented__btn row row--tight",
        type: "button",
        "aria-pressed": String(state.theme === option.value),
      },
      icon(option.iconName, { size: 14 }),
      el("span", { text: option.label })
    );
    button.addEventListener("click", () => {
      setTheme(option.value);
      buttons.forEach((other, index) =>
        other.setAttribute("aria-pressed", String(OPTIONS[index].value === option.value))
      );
    });
    return button;
  });

  group.append(...buttons);

  return el(
    "section",
    { class: "card stack stack--sm" },
    el("div", { class: "card__head" }, el("h2", { class: "card__title", text: "Appearance" })),
    el(
      "div",
      { class: "settings-row" },
      el(
        "div",
        { class: "settings-row__text" },
        el("div", { class: "settings-row__title", text: "Theme" }),
        el("div", {
          class: "settings-row__note",
          text: "System follows your device. Applied before first paint, so there is no flash on load.",
        })
      ),
      el("div", { class: "settings-row__control" }, group)
    )
  );
}

/* ----------------------------------------------------------------- Instance */

const MODEL_LABELS = {
  detection: "Detection",
  depth: "Depth",
  classification: "Classification",
  nutrition: "Nutrition",
};

/** What is actually running. Without trained weights the pipeline falls back to
 *  classical CV, and a user comparing two numbers deserves to know which engine
 *  produced them. */
function instanceCard() {
  const health = state.health;
  if (!health) return null;

  const full = health.engine === "full";

  return el(
    "section",
    { class: "card stack stack--sm" },
    el(
      "div",
      { class: "card__head" },
      el("h2", { class: "card__title", text: "This instance" }),
      el("span", {
        class: full ? "chip chip--brand" : "chip chip--warn",
        text: full ? "Trained weights" : "Built-in estimator",
      })
    ),
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
            el("th", { text: "Stage" }),
            el("th", { text: "Backend" }),
            el("th", { text: "Version" })
          )
        ),
        el(
          "tbody",
          {},
          Object.entries(health.models || {}).map(([key, info]) =>
            el(
              "tr",
              {},
              el("td", { text: MODEL_LABELS[key] || key }),
              el("td", {}, el("code", { class: "step__model", text: info.backend || "—" })),
              el("td", { class: "faint", text: info.version || "—" })
            )
          )
        )
      )
    ),
    el(
      "div",
      { class: "row row--tight" },
      el("span", { class: "chip", text: `v${health.version}` }),
      el("span", { class: "chip", text: health.database }),
      el("span", { class: "chip", text: `≤ ${Math.round((health.limits?.max_upload_bytes || 0) / 1048576)} MB uploads` }),
      el("span", { class: "chip", text: `≤ ${health.limits?.max_items_per_plate ?? "—"} items` })
    ),
    full
      ? null
      : el("p", {
          class: "xsmall faint",
          text: "Trained checkpoints aren't installed, so detection, depth and classification use their geometric fallbacks. Labels are best treated as a starting point; the measurement maths is identical either way.",
        }),
    el("a", { class: "textlink", href: "/method", text: "How the estimate is made →" })
  );
}

/* ------------------------------------------------------------------ Session */

function sessionCard() {
  const guest = isGuest();

  const signOut = el(
    "button",
    { class: "btn btn--outline" },
    icon("logout", { size: 16 }),
    el("span", { text: guest ? "End guest session" : "Sign out" })
  );

  signOut.addEventListener("click", async () => {
    const ok = await confirmAction({
      title: guest ? "End this guest session?" : "Sign out?",
      message: guest
        ? "Guest meals are reachable only through this session's token. Ending it means they cannot be opened again — there is no email to sign back in with."
        : "Your meals and goals stay on the server. Signing back in brings them back.",
      confirmLabel: guest ? "End session" : "Sign out",
      danger: guest,
    });
    if (!ok) return;

    clearSession();
    try {
      // Straight into a fresh guest session, so the upload flow still works on
      // the very next click rather than erroring on a missing token.
      setSession(await api.guest());
    } catch {
      /* Offline: bootSession will retry on the next load. */
    }
    toastOk(guest ? "Guest session ended." : "Signed out.");
    navigate("/", { replace: true });
  });

  return el(
    "section",
    { class: "card stack stack--sm" },
    el("div", { class: "card__head" }, el("h2", { class: "card__title", text: "Session and data" })),
    el(
      "div",
      { class: "settings-row" },
      el(
        "div",
        { class: "settings-row__text" },
        el("div", { class: "settings-row__title", text: "Delete a meal" }),
        el("div", {
          class: "settings-row__note",
          text: "Deletion is per meal, from its own page, and it unlinks the photo from disk immediately.",
        })
      ),
      el(
        "div",
        { class: "settings-row__control" },
        el(
          "a",
          { href: "/history", class: "btn btn--ghost btn--sm" },
          icon("history", { size: 15 }),
          el("span", { text: "Open history" })
        )
      )
    ),
    el(
      "div",
      { class: "settings-row" },
      el(
        "div",
        { class: "settings-row__text" },
        el("div", { class: "settings-row__title", text: guest ? "Guest session" : "Signed-in session" }),
        el("div", {
          class: "settings-row__note",
          text: "The token is held in this browser's local storage. Nothing else about you is stored client-side.",
        })
      ),
      el("div", { class: "settings-row__control" }, signOut)
    )
  );
}
