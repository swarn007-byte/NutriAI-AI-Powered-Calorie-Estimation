/* API client. One place that knows about HTTP, tokens, and the error contract
 * in design.md §10.
 *
 * The important design point: every non-2xx response becomes an `ApiError`
 * carrying the *status* alongside the server's own message, because the UI
 * treats the statuses very differently — 422 is "that isn't food", 400 is "that
 * file is wrong", 503 is "the models aren't up". Collapsing them into a generic
 * failure would lose the only information the user can act on.
 */

const BASE = "/api";

export class ApiError extends Error {
  constructor(status, message, detail) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }

  /** Whether retrying the identical request could plausibly succeed. */
  get retryable() {
    return this.status === 0 || this.status === 503 || this.status >= 500;
  }
}

let token = null;
let onUnauthorized = null;

export function setToken(value) {
  token = value || null;
}

export function getToken() {
  return token;
}

/** Registered by the store so a 401 anywhere clears the session exactly once. */
export function setUnauthorizedHandler(fn) {
  onUnauthorized = fn;
}

function authHeaders(extra = {}) {
  return token ? { ...extra, Authorization: `Bearer ${token}` } : { ...extra };
}

async function request(path, { method = "GET", body, headers, signal, raw } = {}) {
  let response;
  try {
    response = await fetch(`${BASE}${path}`, {
      method,
      headers: authHeaders(headers),
      body,
      signal,
    });
  } catch (error) {
    if (error?.name === "AbortError") throw error;
    // Status 0 is the offline / DNS / connection-refused case. It is the one
    // failure the user can fix themselves, so it gets its own wording.
    throw new ApiError(0, "Can't reach the server. Check your connection and try again.");
  }

  if (response.status === 401 && onUnauthorized) onUnauthorized();

  if (response.status === 204) return null;

  const isJson = (response.headers.get("content-type") || "").includes("application/json");
  const payload = isJson ? await response.json().catch(() => null) : await response.text();

  if (!response.ok) {
    throw new ApiError(response.status, extractMessage(payload, response.status), payload);
  }

  return raw ? response : payload;
}

/** FastAPI reports HTTPException as `{detail: "..."}` and validation failures as
 *  `{detail: [{loc, msg, ...}]}`. Both must become one readable sentence. */
function extractMessage(payload, status) {
  const detail = payload && typeof payload === "object" ? payload.detail : null;

  if (typeof detail === "string" && detail.trim()) return detail;

  if (Array.isArray(detail) && detail.length) {
    const first = detail[0];
    const field = Array.isArray(first?.loc) ? first.loc[first.loc.length - 1] : null;
    const message = first?.msg || "is invalid";
    const cleaned = String(message).replace(/^Value error,\s*/i, "");
    return field && field !== "body" ? `${labelFor(field)}: ${cleaned}` : cleaned;
  }

  if (typeof payload === "string" && payload.trim() && payload.length < 200) return payload;

  return FALLBACK[status] || `Request failed (${status}).`;
}

const labelFor = (field) =>
  String(field)
    .replace(/_/g, " ")
    .replace(/^\w/, (c) => c.toUpperCase());

const FALLBACK = {
  400: "That image couldn't be used.",
  401: "Please sign in again.",
  403: "You don't have access to that.",
  404: "That doesn't exist any more.",
  409: "That already exists.",
  413: "That file is too large.",
  422: "No food could be found in that photo.",
  429: "Too many requests — give it a moment.",
  503: "The analysis models aren't loaded yet.",
};

const json = (path, method, body) =>
  request(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });

/* ------------------------------------------------------------------- Routes */

export const api = {
  health: () => request("/health"),

  catalog: () => request("/nutrition/catalog"),

  lookup: (food) => request(`/nutrition/lookup?food=${encodeURIComponent(food)}`),

  guest: () => json("/auth/guest", "POST"),

  register: (payload) => json("/auth/register", "POST", payload),

  login: (payload) => json("/auth/login", "POST", payload),

  me: () => request("/auth/me"),

  savePreferences: (payload) => json("/users/me/preferences", "PATCH", payload),

  /** The one-shot analysis: photo and plate size in, finished meal out.
   *
   *  No screen calls this any more — the UI goes through `scan` then
   *  `analyzeDraft`, so the user confirms the item list before anything is
   *  costed. The endpoint is still live and still tested, because it is the
   *  whole API for a caller that has no user to ask, so the client keeps it.
   *
   *  Content-Type is deliberately unset — the browser must add the multipart
   *  boundary itself, and setting it by hand breaks parsing. */
  analyze: ({ file, plateDiameterCm, notes, signal }) => {
    const form = new FormData();
    form.append("image", file, file.name || "meal.jpg");
    if (plateDiameterCm) form.append("plate_diameter_cm", String(plateDiameterCm));
    if (notes) form.append("notes", notes);
    return request("/meals/analyze", { method: "POST", body: form, signal });
  },

  /** Phase one: what is on the plate. No plate size, no nutrition — the point is
   *  to come back fast enough that the user is still looking at the photo. Same
   *  deliberate absence of Content-Type as `analyze`. */
  scan: ({ file, notes, signal }) => {
    const form = new FormData();
    form.append("image", file, file.name || "meal.jpg");
    if (notes) form.append("notes", notes);
    return request("/meals/scan", { method: "POST", body: form, signal });
  },

  /** Phase two: cost the plate the user just confirmed. Consumes the draft. */
  analyzeDraft: (draftId, payload, { signal } = {}) =>
    request(`/meals/${encodeURIComponent(draftId)}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    }),

  manualMeal: (payload) => json("/meals/manual", "POST", payload),

  meal: (id) => request(`/meals/${encodeURIComponent(id)}`),

  deleteMeal: (id) => request(`/meals/${encodeURIComponent(id)}`, { method: "DELETE" }),

  correctItem: (mealId, itemId, payload) =>
    json(`/meals/${encodeURIComponent(mealId)}/items/${encodeURIComponent(itemId)}`, "PATCH", payload),

  adjustPlate: (mealId, plateDiameterCm) =>
    json(`/meals/${encodeURIComponent(mealId)}/plate`, "PATCH", {
      plate_diameter_cm: plateDiameterCm,
    }),

  history: ({ limit = 30, offset = 0 } = {}) =>
    request(`/users/me/history?limit=${limit}&offset=${offset}`),

  /** The server buckets meals into local days, so it needs the client's offset.
   *  getTimezoneOffset() is minutes *behind* UTC, hence the negation. */
  summary: ({ days = 14 } = {}) =>
    request(`/users/me/summary?days=${days}&tz_offset=${-new Date().getTimezoneOffset()}`),
};
