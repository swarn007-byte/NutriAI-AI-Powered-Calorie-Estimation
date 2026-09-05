/* Keeps `[data-tip]` tooltips inside the viewport.
 *
 * The tooltips themselves are pure CSS (components.css) — but *placement* is not
 * something CSS can settle. A tooltip centred on its anchor needs half its own
 * width of room to each side, and whether that room exists depends on where the
 * anchor happens to land at the current viewport width. Tagging an alignment by
 * hand holds only until the row wraps or the copy changes length; it has already
 * failed here three times, on the top bar, the trend chart and the meal header.
 *
 * Hover and focus are the only two triggers, so there is a free moment to measure
 * just before the box appears: flip the anchor to whichever edge keeps it on
 * screen, and let CSS do the rest.
 */

const hoverable = matchMedia("(hover: hover) and (pointer: fine)");

function place(host) {
  // Top-bar tooltips are anchored to the bar rather than to the control, which
  // gives them the full gutter-to-gutter width at any viewport — nothing to fix,
  // and that rule outranks the alignment helpers anyway.
  if (host.closest(".topbar")) return;

  host.removeAttribute("data-tip-align");

  // The pseudo-element is hidden with `opacity`, not `display`, so its used width
  // is already resolved and readable before it is ever shown.
  const width = parseFloat(getComputedStyle(host, "::after").width);
  if (!Number.isFinite(width)) return;

  const rect = host.getBoundingClientRect();
  const centre = rect.left + rect.width / 2;
  const gutter = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--gutter")) || 16;
  const viewport = document.documentElement.clientWidth;

  if (centre - width / 2 < gutter) host.setAttribute("data-tip-align", "start");
  else if (centre + width / 2 > viewport - gutter) host.setAttribute("data-tip-align", "end");
}

export function tooltips() {
  if (!hoverable.matches) return;
  const handler = (event) => {
    const host = event.target.closest?.("[data-tip]");
    if (host) place(host);
  };
  // Capture, because `focus` does not bubble and delegated `pointerover` should
  // still fire for hosts whose children stop propagation.
  document.addEventListener("pointerover", handler, true);
  document.addEventListener("focusin", handler, true);
}
