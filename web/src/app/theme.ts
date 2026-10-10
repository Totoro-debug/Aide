
export type Theme = "system" | "light" | "dark";

const THEME_KEY = "aide.theme";

export function readThemePreference(): Theme {
  try {
    const value = window.localStorage.getItem(THEME_KEY);
    if (value === "light" || value === "dark" || value === "system") return value;
  } catch {
    // Theme preference is optional when storage is unavailable.
  }
  return "light";
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme;
  try {
    window.localStorage.setItem(THEME_KEY, theme);
  } catch {
    // Theme preference is optional when storage is unavailable.
  }
}
