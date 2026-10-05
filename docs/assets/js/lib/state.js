// URL-hash state: "#/<experiment>/<view>?key=value&...". Every view reads from here,
// so any screen can be shared or bookmarked as a link.

export function parseHash() {
  const raw = location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const [exp, view] = path.split("/").filter(Boolean);
  return { exp, view, params: Object.fromEntries(new URLSearchParams(query)) };
}

export function hrefFor(exp, view, params = {}) {
  const q = new URLSearchParams(Object.entries(params).filter(([, v]) => v != null && v !== ""));
  const qs = q.toString().replace(/%2C/g, ",");
  return `#/${[exp, view].filter(Boolean).join("/")}${qs ? "?" + qs : ""}`;
}

/** Merge `patch` into the current params (null deletes) and re-render without adding history. */
export function setParams(patch) {
  const { exp, view, params } = parseHash();
  const next = { ...params };
  for (const [k, v] of Object.entries(patch)) {
    if (v == null || v === "") delete next[k];
    else next[k] = String(v);
  }
  history.replaceState(null, "", hrefFor(exp, view, next));
  window.dispatchEvent(new Event("statechange"));
}

export const store = {
  get(key, fallback = null) {
    try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem(key, value); } catch { /* storage unavailable */ }
  },
};
