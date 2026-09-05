/* Toasts and confirm dialogs — the two things every page needs and nobody
 * should reimplement.
 *
 * Toasts are `role="status"` in a live region so screen readers announce them
 * without stealing focus. Errors are `role="alert"`, which interrupts, because
 * an error the user does not hear about is an error they cannot act on.
 */

import { el, icon, mount } from "./dom.js";

let stack = null;

function ensureStack() {
  if (stack?.isConnected) return stack;
  stack = el("div", {
    class: "toast-stack",
    "aria-live": "polite",
    "aria-atomic": "false",
  });
  document.body.appendChild(stack);
  return stack;
}

const ICONS = { ok: "check", warn: "alert", error: "alert", info: "info" };

export function toast(message, { title, kind = "info", duration } = {}) {
  const host = ensureStack();
  const ms = duration ?? (kind === "error" ? 6500 : 3800);

  const node = el(
    "div",
    {
      class: ["toast", `toast--${kind}`],
      role: kind === "error" ? "alert" : "status",
    },
    icon(ICONS[kind] || "info"),
    el(
      "div",
      { class: "toast__body" },
      title ? el("div", { class: "toast__title", text: title }) : null,
      el("div", { class: title ? "toast__text" : "toast__title", text: message })
    ),
    el(
      "button",
      { class: "btn btn--ghost btn--icon btn--sm", "aria-label": "Dismiss", onclick: () => dismiss(node) },
      icon("x", { size: 15 })
    )
  );

  host.appendChild(node);

  // Cap the stack; three simultaneous toasts is already a design failure, and
  // more than that buries the newest one.
  while (host.children.length > 3) dismiss(host.firstElementChild);

  const timer = setTimeout(() => dismiss(node), ms);
  // Pausing on hover lets a user actually read a long error message.
  node.addEventListener("mouseenter", () => clearTimeout(timer));
  return node;
}

function dismiss(node) {
  if (!node || node.classList.contains("toast--out")) return;
  node.classList.add("toast--out");
  node.addEventListener("animationend", () => node.remove(), { once: true });
  // Belt and braces: if the animation is suppressed the event still fires
  // (0.01ms duration), but a detached node would never emit it.
  setTimeout(() => node.remove(), 600);
}

export const toastOk = (message, options) => toast(message, { ...options, kind: "ok" });
export const toastWarn = (message, options) => toast(message, { ...options, kind: "warn" });
export const toastError = (message, options) => toast(message, { ...options, kind: "error" });

/* ------------------------------------------------------------------- Sheets */

/** Open a modal sheet. Returns a `close` function.
 *
 *  Focus is trapped while open and returned to the invoking element on close —
 *  without that, a keyboard user tabs into the page behind the scrim and has no
 *  way back.
 */
export function openSheet({ title, body, footer, onClose, labelledBy = "sheet-title" } = {}) {
  const previous = document.activeElement;
  const scrim = el("div", { class: "scrim" });

  const sheet = el(
    "div",
    { class: "sheet", role: "dialog", "aria-modal": "true", "aria-labelledby": labelledBy },
    el("div", { class: "sheet__grip" }),
    el(
      "div",
      { class: "sheet__head" },
      el("h2", { class: "sheet__title", id: labelledBy, text: title || "" }),
      el(
        "button",
        { class: "btn btn--ghost btn--icon btn--sm", "aria-label": "Close", onclick: () => close() },
        icon("x", { size: 17 })
      )
    ),
    el("div", { class: "sheet__body" }),
    footer ? el("div", { class: "sheet__foot" }) : null
  );

  if (body) mount(sheet.querySelector(".sheet__body"), body);
  if (footer) mount(sheet.querySelector(".sheet__foot"), footer);

  let closed = false;
  function close(result) {
    if (closed) return;
    closed = true;
    document.removeEventListener("keydown", onKey, true);
    scrim.remove();
    sheet.remove();
    document.body.style.removeProperty("overflow");
    if (previous instanceof HTMLElement) previous.focus({ preventScroll: true });
    onClose?.(result);
  }

  function onKey(event) {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      return;
    }
    if (event.key !== "Tab") return;

    const focusable = sheet.querySelectorAll(
      'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
    );
    if (!focusable.length) return;
    const first = focusable[0];
    const last = focusable[focusable.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }

  scrim.addEventListener("click", () => close());
  document.addEventListener("keydown", onKey, true);
  document.body.style.overflow = "hidden";
  document.body.append(scrim, sheet);

  const target = sheet.querySelector("[data-autofocus]") || sheet.querySelector(".sheet__body button, .sheet__body input");
  (target || sheet.querySelector(".sheet__head button")).focus({ preventScroll: true });

  return { close, sheet };
}

/** Promise-based confirm. Used for meal deletion, which is irreversible —
 *  the image is unlinked from disk, not soft-deleted (main.py:441). */
export function confirmAction({
  title = "Are you sure?",
  message,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  danger = false,
} = {}) {
  return new Promise((resolve) => {
    let settled = false;
    const finish = (value) => {
      if (settled) return;
      settled = true;
      resolve(value);
    };

    const confirm = el("button", {
      class: ["btn", danger ? "btn--danger" : "btn--primary"],
      text: confirmLabel,
      "data-autofocus": "true",
    });
    const cancel = el("button", { class: "btn btn--ghost", text: cancelLabel });

    const { close } = openSheet({
      title,
      body: el("p", { class: "muted", text: message || "" }),
      footer: [cancel, confirm],
      onClose: () => finish(false),
    });

    confirm.addEventListener("click", () => {
      finish(true);
      close();
    });
    cancel.addEventListener("click", () => close());
  });
}
