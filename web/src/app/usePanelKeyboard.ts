import { useEffect } from "react";

export function usePanelKeyboard(open: boolean, setOpen: (open: boolean) => void, panelId: string, triggerId: string) {
  useEffect(() => {
    if (!open || !window.matchMedia("(max-width: 1024px)").matches) return;
    const panel = document.getElementById(panelId);
    const trigger = document.getElementById(triggerId);
    if (panel === null) return;
    const controls = () => Array.from(panel.querySelectorAll<HTMLElement>(
      'a[href], button:not(:disabled), input:not(:disabled), [tabindex="0"]',
    )).filter((element) => element.getClientRects().length > 0);
    controls()[0]?.focus();
    const handleKey = (event: KeyboardEvent) => {
      if (!window.matchMedia("(max-width: 1024px)").matches || panel.getClientRects().length === 0) return;
      if (event.defaultPrevented || document.querySelector('[role="dialog"]') !== null) return;
      if (event.key === "Escape") {
        event.preventDefault();
        trigger?.focus();
        setOpen(false);
      } else if (event.key === "Tab") {
        const items = controls();
        const first = items[0];
        const last = items.at(-1);
        if (!panel.contains(document.activeElement)
          || (event.shiftKey ? document.activeElement === first : document.activeElement === last)) {
          event.preventDefault();
          (event.shiftKey ? last : first)?.focus();
        }
      }
    };
    window.addEventListener("keydown", handleKey);
    return () => {
      window.removeEventListener("keydown", handleKey);
      if (panel.contains(document.activeElement) || document.activeElement === document.body) trigger?.focus();
    };
  }, [open, setOpen, panelId, triggerId]);
}
