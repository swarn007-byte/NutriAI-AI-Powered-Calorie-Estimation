/* Application state. A single observable object with a pub/sub notify — the app
 * has one user, one active meal and a handful of caches, so a reducer/action
 * architecture would be ceremony without benefit.
 *
 * Persistence rules, which are the only subtle part:
 *   - the token and the theme go to localStorage (they must survive a reload)
 *   - the active meal goes to sessionStorage (a reload should not resurrect
 *     yesterday's result as if it were current, but an accidental back/forward
 *     within the session should not lose it)
 *   - nothing else is persisted; history and summary are always refetched,
 *     because a stale calorie total is worse than a spinner
 */

import { api, setToken, setUnauthorizedHandler } from "./api.js";

const TOKEN_KEY = "nutriai.token";
const THEME_KEY = "nutriai.theme";
const MEAL_KEY = "nutriai.meal";
const DRAFT_KEY = "nutriai.draft";

/** localStorage throws in Safari private mode and when quota is exhausted. It
 *  is a cache, never a source of truth, so every access is best-effort. */
const safe = {
  get(store, key) {
    try {
      return store.getItem(key);
    } catch {
      return null;
    }
  },
  set(store, key, value) {
    try {
      if (value === null || value === undefined) store.removeItem(key);
      else store.setItem(key, value);
    } catch {
      /* ignore */
    }
  },
};

export const state = {
  user: null,
  token: safe.get(localStorage, TOKEN_KEY),
  theme: safe.get(localStorage, THEME_KEY) || "system",
  health: null,
  catalog: null,
  meal: null,
  draft: null,
  history: null,
  summary: null,
  booting: true,
};

setToken(state.token);

const listeners = new Set();

export function subscribe(listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

function notify(keys) {
  for (const listener of listeners) listener(keys);
}

/** The only mutation path. Returns the changed keys so listeners can bail early. */
export function set(patch) {
  const changed = [];
  for (const [key, value] of Object.entries(patch)) {
    if (state[key] !== value) {
      state[key] = value;
      changed.push(key);
    }
  }
  if (changed.length) notify(changed);
  return changed;
}

/* -------------------------------------------------------------------- Theme */

let mediaQuery = null;

export function applyTheme(theme = state.theme) {
  const resolved = theme === "system" ? systemTheme() : theme;
  document.documentElement.dataset.theme = resolved;

  // Keep the mobile browser chrome in step with the app background, or the
  // status bar sits as a bright band above a dark app.
  const meta = document.querySelector('meta[name="theme-color"]');
  if (meta) meta.setAttribute("content", resolved === "light" ? "#f5f8f5" : "#07100c");

  // Only follow the OS while the user has actually chosen "system".
  if (theme === "system" && !mediaQuery) {
    mediaQuery = window.matchMedia("(prefers-color-scheme: light)");
    mediaQuery.addEventListener("change", () => {
      if (state.theme === "system") applyTheme("system");
    });
  }
}

const systemTheme = () =>
  window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";

export function setTheme(theme) {
  set({ theme });
  safe.set(localStorage, THEME_KEY, theme);
  applyTheme(theme);
  // Remember it server-side too, so the choice follows a signed-in user to
  // another device. Failure here is invisible and harmless.
  if (state.user && !state.user.is_guest) api.savePreferences({ theme }).catch(() => {});
}

/* ------------------------------------------------------------------ Session */

export function setSession({ token, user }) {
  set({ token: token ?? state.token, user: user ?? state.user });
  if (token) {
    setToken(token);
    safe.set(localStorage, TOKEN_KEY, token);
  }
  // A signed-in user's stored theme preference wins over the local one, unless
  // they have explicitly set one on this device since.
  const remote = user?.preferences?.theme;
  if (remote && !safe.get(localStorage, THEME_KEY)) {
    set({ theme: remote });
    applyTheme(remote);
  }
}

export function clearSession() {
  setToken(null);
  safe.set(localStorage, TOKEN_KEY, null);
  safe.set(sessionStorage, MEAL_KEY, null);
  safe.set(sessionStorage, DRAFT_KEY, null);
  set({ token: null, user: null, meal: null, draft: null, history: null, summary: null });
}

setUnauthorizedHandler(() => {
  // Only clear if we thought we were authenticated; otherwise a probe request
  // on a fresh load would wipe a token that is merely unvalidated.
  if (state.token) clearSession();
});

/** Resolve who we are. Falls back to a guest session so the upload flow is
 *  never gated behind a sign-up — that is the core product promise. */
export async function bootSession() {
  if (state.token) {
    try {
      const user = await api.me();
      setSession({ user });
      return user;
    } catch {
      clearSession();
    }
  }
  try {
    const payload = await api.guest();
    setSession(payload);
    return payload.user;
  } catch {
    // Offline on first load. The UI stays usable and read-only; the next
    // action will retry and surface a real error.
    return null;
  }
}

export const isGuest = () => !state.user || state.user.is_guest === true;

export const displayName = () =>
  state.user?.name || (state.user?.email ? state.user.email.split("@")[0] : "Guest");

export function preference(key, fallback) {
  const value = state.user?.preferences?.[key];
  return value === undefined || value === null ? fallback : value;
}

/* --------------------------------------------------------------- Active meal */

export function setMeal(meal) {
  set({ meal });
  safe.set(sessionStorage, MEAL_KEY, meal ? JSON.stringify(meal) : null);
  // A new meal invalidates both aggregates. Null rather than a refetch: the
  // pages that need them fetch on entry, and this may be called from anywhere.
  set({ history: null, summary: null });
}

export function restoreMeal() {
  const raw = safe.get(sessionStorage, MEAL_KEY);
  if (!raw) return null;
  try {
    const meal = JSON.parse(raw);
    if (meal?.meal_id) {
      set({ meal });
      return meal;
    }
  } catch {
    safe.set(sessionStorage, MEAL_KEY, null);
  }
  return null;
}

/* ------------------------------------------------- Draft (the review step) */

/* Unlike `pending`, a draft is plain JSON — the server holds the photo and the
 * region masks, and the client only carries ids and labels. So it survives a
 * refresh on /review, which matters: the user may be squinting at the photo,
 * comparing it to the list, for a while. sessionStorage rather than local for
 * the same reason as the meal — a draft from yesterday is not current, and its
 * server-side row has been swept anyway. */

export function setDraft(draft) {
  set({ draft });
  safe.set(sessionStorage, DRAFT_KEY, draft ? JSON.stringify(draft) : null);
}

export function restoreDraft() {
  const raw = safe.get(sessionStorage, DRAFT_KEY);
  if (!raw) return null;
  try {
    const draft = JSON.parse(raw);
    if (draft?.draft_id) {
      set({ draft });
      return draft;
    }
  } catch {
    safe.set(sessionStorage, DRAFT_KEY, null);
  }
  return null;
}

/* ------------------------------------------------------------------- Caches */

/** Health and catalog are effectively static for a session; fetch once. */
export async function ensureHealth() {
  if (state.health) return state.health;
  const health = await api.health();
  set({ health });
  return health;
}

export async function ensureCatalog() {
  if (state.catalog) return state.catalog;
  const catalog = await api.catalog();
  set({ catalog });
  return catalog;
}

export function limit(key, fallback) {
  const value = state.health?.limits?.[key];
  return value === undefined || value === null ? fallback : value;
}
