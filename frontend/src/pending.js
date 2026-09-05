/* The one piece of state that cannot live in the store.
 *
 * A `File` is not serialisable, so the chosen image cannot go through
 * sessionStorage like the rest of the app's state. It is handed from the upload
 * page to the processing page through this module and consumed exactly once —
 * `take()` clears it, so a refresh on /analyzing bounces home instead of
 * re-running an analysis the user did not ask for a second time.
 */

let pending = null;

export function setPending(next) {
  // Only revoke when the blob is actually being replaced. home.js calls this a
  // second time on submit to attach the plate width and note, carrying the same
  // previewUrl through — revoking unconditionally handed /analyzing a dead URL
  // and left the photo blank for the whole wait.
  if (pending?.previewUrl && pending.previewUrl !== next?.previewUrl) {
    URL.revokeObjectURL(pending.previewUrl);
  }
  pending = next;
}

export function peekPending() {
  return pending;
}

export function takePending() {
  const value = pending;
  pending = null;
  return value;
}

/** Revoke the preview URL. Object URLs pin the whole blob in memory until
 *  revoked, and a 10 MB photo per abandoned upload adds up quickly. */
export function release() {
  if (pending?.previewUrl) URL.revokeObjectURL(pending.previewUrl);
  pending = null;
}
