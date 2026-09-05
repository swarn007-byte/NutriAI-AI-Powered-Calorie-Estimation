/* Auth — sign in, create an account, or carry on as a guest.
 *
 * This page is never a gate. The upload flow works on a guest token from the
 * first visit, so the honest framing here is "keep your meals", not "sign up to
 * continue". Registering while holding a guest token *upgrades that same row*
 * (main.py:264), so the meals already analysed carry over — which is worth
 * saying out loud, because users assume the opposite.
 */

import { el, icon, mount } from "../dom.js";
import { navigate } from "../router.js";
import { api, ApiError } from "../api.js";
import { setSession, isGuest, state, displayName } from "../store.js";
import { toastOk, toastError } from "../toast.js";

export default function auth({ query } = {}) {
  const page = el("div", { class: "page" });
  const host = el("div", { class: "auth" });
  page.append(host);

  // Where to go afterwards. Same-origin paths only, so a crafted `?next=` can't
  // turn a sign-in into an open redirect.
  const next = safeNext(query?.next);

  if (state.user && !isGuest()) {
    host.replaceChildren(alreadyIn(next));
    return page;
  }

  /* Switching mode re-renders, so the typed values live out here — being bounced
   * from "create account" to "sign in" by a 409 and losing the email you just
   * typed is exactly the moment a user gives up. */
  const draft = { name: "", email: "", password: "" };
  let mode = query?.mode === "register" ? "register" : "signin";

  const setMode = (to) => {
    if (to === mode) return;
    mode = to;
    paint();
  };

  function paint() {
    // mount(), not replaceChildren(): the name field and the carry-over note are
    // both absent in sign-in mode, and a null child would render as "null".
    mount(host, ...form({ mode, next, draft, setMode }));
  }

  paint();
  return page;
}

const safeNext = (value) =>
  typeof value === "string" && value.startsWith("/") && !value.startsWith("//") ? value : "/today";

/* --------------------------------------------------------------------- Form */

function form({ mode, next, draft, setMode }) {
  const register = mode === "register";

  const nameField = register
    ? field({
        label: "Name",
        type: "text",
        name: "name",
        autocomplete: "name",
        placeholder: "Optional",
        value: draft.name,
      })
    : null;

  const emailField = field({
    label: "Email",
    type: "email",
    name: "email",
    autocomplete: "email",
    inputmode: "email",
    placeholder: "you@example.com",
    value: draft.email,
  });

  const passwordField = passwordInput({
    autocomplete: register ? "new-password" : "current-password",
    hint: register ? "At least 8 characters." : null,
    value: draft.password,
  });

  const strength = register ? strengthMeter() : null;
  if (strength) {
    passwordField.input.addEventListener("input", () => strength.set(passwordField.input.value));
    strength.set(draft.password);
  }

  // Keep the draft in step so a mode switch preserves what was typed.
  const remember = () => {
    draft.name = nameField ? nameField.input.value : draft.name;
    draft.email = emailField.input.value;
    draft.password = passwordField.input.value;
  };
  emailField.input.addEventListener("input", remember);
  passwordField.input.addEventListener("input", remember);
  nameField?.input.addEventListener("input", remember);

  const submit = el(
    "button",
    { class: "btn btn--primary btn--lg btn--block", type: "submit" },
    icon(register ? "sparkles" : "login", { size: 17 }),
    el("span", { text: register ? "Create account" : "Sign in" })
  );

  const node = el(
    "form",
    { class: "auth__form", novalidate: true },
    nameField?.node,
    emailField.node,
    passwordField.node,
    strength?.node,
    submit
  );

  // Native validation is suppressed (`novalidate`) so the messages match the
  // server's wording instead of sitting in a bubble the styling can't reach.
  let pending = false;
  node.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (pending) return;

    emailField.clearError();
    passwordField.clearError();
    remember();

    const email = draft.email.trim();
    const password = draft.password;

    if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
      emailField.setError("Enter a valid email address.");
      emailField.input.focus();
      return;
    }
    if (!password) {
      passwordField.setError("Enter your password.");
      passwordField.input.focus();
      return;
    }
    if (register && password.length < 8) {
      passwordField.setError("Use at least 8 characters.");
      passwordField.input.focus();
      return;
    }

    pending = true;
    submit.disabled = true;
    submit.setAttribute("aria-busy", "true");

    try {
      const payload = register
        ? await api.register({ email, password, name: draft.name.trim() || null })
        : await api.login({ email, password });
      setSession(payload);
      draft.password = "";
      toastOk(
        register
          ? `Account created. Welcome, ${payload.user?.name || "you"}.`
          : `Signed in as ${payload.user?.email || email}.`
      );
      navigate(next, { replace: true });
      return;
    } catch (error) {
      const status = error instanceof ApiError ? error.status : 0;
      if (status === 409) {
        draft.password = "";
        setMode("signin");
        toastError("That email already has an account — sign in instead.", {
          title: "Already registered",
        });
        return;
      }
      if (status === 401) {
        passwordField.setError("Email or password is incorrect.");
        passwordField.input.focus();
        return;
      }
      toastError(error?.message || "That didn't work.", {
        title: register ? "Couldn't create the account" : "Couldn't sign you in",
      });
    } finally {
      pending = false;
      submit.disabled = false;
      submit.removeAttribute("aria-busy");
    }
  });

  return [
    el(
      "div",
      { class: "auth__head" },
      el(
        "div",
        { class: "center", style: { marginBottom: "0.4rem" } },
        icon(register ? "sparkles" : "login", { size: 30 })
      ),
      el("h1", { style: { fontSize: "var(--step-3)" }, text: register ? "Keep your meals" : "Welcome back" }),
      el("p", {
        class: "muted small",
        text: register
          ? "An account keeps your history and goals, and makes them reachable from another device."
          : "Sign in to pick up your history and goals.",
      })
    ),
    el(
      "div",
      { class: "center" },
      el(
        "div",
        { class: "segmented", role: "group", "aria-label": "Sign in or create an account" },
        segButton("Sign in", !register, () => setMode("signin")),
        segButton("Create account", register, () => setMode("register"))
      )
    ),
    node,
    register && isGuest() && state.user
      ? el(
          "div",
          { class: "panel panel--info" },
          el(
            "div",
            { class: "row row--tight", style: { marginBottom: "0.3rem" } },
            icon("info", { size: 16 }),
            el("strong", { text: "Your guest meals come with you" })
          ),
          el("p", {
            class: "small muted",
            text: "Everything analysed in this session stays attached to the new account — nothing is re-uploaded and nothing is lost.",
          })
        )
      : null,
    el("div", { class: "divider", text: "or" }),
    guestBlock(next),
  ];
}

function segButton(label, active, onClick) {
  const button = el("button", {
    class: "segmented__btn",
    type: "button",
    "aria-pressed": String(active),
    text: label,
  });
  button.addEventListener("click", onClick);
  return button;
}

/* ------------------------------------------------------------------- Fields */

function field({ label, hint, ...attrs }) {
  const input = el("input", { class: "input", ...attrs });
  const error = el("div", { class: "field__error", hidden: true });
  const node = el(
    "label",
    { class: "field" },
    el("span", { class: "field__label", text: label }),
    input,
    hint ? el("span", { class: "field__hint", text: hint }) : null,
    error
  );
  return { node, input, ...errorApi(input, error) };
}

/** Password field with a reveal toggle. Typing a password blind on a phone
 *  keyboard is the biggest single cause of a failed sign-in.
 *
 *  Not a `<label>` wrapper like the others: nesting the toggle button inside a
 *  label makes every tap on it activate the label too. Explicit `for`/`id`. */
let seq = 0;

function passwordInput({ autocomplete, hint, value }) {
  const id = `pw-${(seq += 1)}`;
  const input = el("input", {
    id,
    class: "input",
    type: "password",
    name: "password",
    autocomplete,
    value: value || "",
    style: { paddingRight: "3.1rem" },
  });

  const toggle = el("button", {
    class: "btn btn--ghost btn--icon btn--sm",
    type: "button",
    "aria-label": "Show password",
    style: { position: "absolute", right: "4px", top: "50%", transform: "translateY(-50%)" },
  });
  toggle.append(icon("eye", { size: 17 }));
  toggle.addEventListener("click", () => {
    const shown = input.type === "text";
    input.type = shown ? "password" : "text";
    toggle.setAttribute("aria-label", shown ? "Show password" : "Hide password");
    toggle.replaceChildren(icon(shown ? "eye" : "eyeOff", { size: 17 }));
    input.focus();
  });

  const error = el("div", { class: "field__error", hidden: true });
  const node = el(
    "div",
    { class: "field" },
    el("label", { class: "field__label", for: id, text: "Password" }),
    el("div", { style: { position: "relative" } }, input, toggle),
    hint ? el("span", { class: "field__hint", text: hint }) : null,
    error
  );

  return { node, input, ...errorApi(input, error) };
}

function errorApi(input, error) {
  return {
    setError(message) {
      error.textContent = message;
      error.hidden = false;
      input.setAttribute("aria-invalid", "true");
    },
    clearError() {
      error.hidden = true;
      input.removeAttribute("aria-invalid");
    },
  };
}

/* ---------------------------------------------------------------- Strength */

/** Four segments driven by length and character variety.
 *
 *  Deliberately not a "must contain a symbol" gate: the server's only rule is
 *  eight characters, and a meter that blocks a long passphrase for lacking
 *  punctuation trains worse passwords, not better ones. This informs, it does
 *  not veto — the submit button never depends on it.
 */
const LEVELS = ["", "Too short", "Fair", "Good", "Strong"];
const COLORS = ["var(--brand)", "var(--danger)", "var(--warn)", "var(--brand)", "var(--brand)"];

function strengthMeter() {
  const segments = Array.from({ length: 4 }, () => el("div", { class: "strength__seg" }));
  const caption = el("span", { class: "field__hint", text: "" });

  const node = el(
    "div",
    { class: "stack", style: { gap: "0.3rem" } },
    el("div", { class: "strength" }, segments),
    caption
  );

  return {
    node,
    set(value) {
      const score = value ? scorePassword(value) : 0;
      segments.forEach((segment, index) => {
        segment.classList.toggle("is-on", index < score);
        segment.style.setProperty("--c", COLORS[score]);
      });
      caption.textContent = LEVELS[score];
    },
  };
}

function scorePassword(value) {
  if (value.length < 8) return 1;
  const variety =
    (/[a-z]/.test(value) ? 1 : 0) +
    (/[A-Z]/.test(value) ? 1 : 0) +
    (/\d/.test(value) ? 1 : 0) +
    (/[^A-Za-z0-9]/.test(value) ? 1 : 0);
  if (value.length >= 16 || (value.length >= 12 && variety >= 3)) return 4;
  if (value.length >= 12 || variety >= 3) return 3;
  return 2;
}

/* ------------------------------------------------------------------- Guest */

function guestBlock(next) {
  const button = el(
    "button",
    { class: "btn btn--outline btn--block", type: "button" },
    icon("user", { size: 16 }),
    el("span", { text: state.user ? "Carry on as a guest" : "Continue as a guest" })
  );

  button.addEventListener("click", async () => {
    if (state.user) {
      navigate(next, { replace: true });
      return;
    }
    button.disabled = true;
    try {
      setSession(await api.guest());
      navigate(next, { replace: true });
    } catch (error) {
      toastError(error?.message || "Couldn't start a guest session.");
      button.disabled = false;
    }
  });

  return el(
    "div",
    { class: "stack stack--sm" },
    button,
    el("p", {
      class: "xsmall faint center",
      text: "A guest session works fully — analysis, history and goals — but it lives in this browser only, and clearing site data ends it.",
    })
  );
}

/* -------------------------------------------------------------- Signed in */

function alreadyIn(next) {
  return el(
    "div",
    { class: "empty" },
    el("div", { class: "empty__icon" }, icon("check", { size: 26 })),
    el("h1", { class: "empty__title", text: `Signed in as ${displayName()}` }),
    el("p", { class: "empty__text", text: state.user?.email || "" }),
    el(
      "div",
      { class: "row", style: { justifyContent: "center" } },
      el("a", { href: next, class: "btn btn--primary" }, el("span", { text: "Continue" })),
      el(
        "a",
        { href: "/settings", class: "btn btn--outline" },
        icon("settings", { size: 16 }),
        el("span", { text: "Settings" })
      )
    )
  );
}
