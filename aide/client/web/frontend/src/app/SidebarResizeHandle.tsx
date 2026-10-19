import { useEffect, useRef } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { useTranslation } from "react-i18next";
import styles from "../shared/styles/controls.module.css";

const WIDTH_KEY = "aide.sidebar.width";
const DEFAULT_WIDTH = 232;
const MIN_WIDTH = 200;
const MAX_WIDTH = 480;
const MIN_CONTENT_WIDTH = 640;

function readWidth(): number {
  try {
    const width = Number(localStorage.getItem(WIDTH_KEY));
    if (Number.isFinite(width) && width >= MIN_WIDTH && width <= MAX_WIDTH) return width;
  } catch {
    // Layout preferences are optional when storage is unavailable.
  }
  return DEFAULT_WIDTH;
}

function saveWidth(width: number) {
  try {
    localStorage.setItem(WIDTH_KEY, String(width));
  } catch {
    // Resizing still works when storage is unavailable.
  }
}

export default function SidebarResizeHandle() {
  const { t } = useTranslation();
  const handleRef = useRef<HTMLDivElement>(null);
  const widthRef = useRef(DEFAULT_WIDTH);
  const dragRef = useRef<{
    pointerId: number;
    startX: number;
    startWidth: number;
    cursor: string;
    userSelect: string;
  } | null>(null);

  function applyWidth(width: number) {
    widthRef.current = width;
    handleRef.current?.parentElement?.style.setProperty("--sidebar-preferred-width", `${width}px`);
  }

  function finishDrag() {
    const drag = dragRef.current;
    if (drag === null) return;
    dragRef.current = null;
    document.body.style.cursor = drag.cursor;
    document.body.style.userSelect = drag.userSelect;
    const handle = handleRef.current;
    handle?.removeAttribute("data-dragging");
    if (handle?.hasPointerCapture(drag.pointerId)) handle.releasePointerCapture(drag.pointerId);
    saveWidth(widthRef.current);
  }

  useEffect(() => {
    applyWidth(readWidth());
    window.addEventListener("blur", finishDrag);
    window.addEventListener("resize", finishDrag);
    return () => {
      finishDrag();
      window.removeEventListener("blur", finishDrag);
      window.removeEventListener("resize", finishDrag);
    };
  }, []);

  function startDrag(event: ReactPointerEvent<HTMLDivElement>) {
    if (event.pointerType !== "mouse" || event.button !== 0 || dragRef.current !== null
      || !window.matchMedia("(min-width: 1024px)").matches) return;
    const sidebar = document.getElementById("app-sidebar");
    if (sidebar === null) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startWidth: sidebar.getBoundingClientRect().width,
      cursor: document.body.style.cursor,
      userSelect: document.body.style.userSelect,
    };
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    event.currentTarget.dataset.dragging = "true";
  }

  function moveDrag(event: ReactPointerEvent<HTMLDivElement>) {
    const drag = dragRef.current;
    if (drag === null || event.pointerId !== drag.pointerId) return;
    const maximum = Math.min(MAX_WIDTH, window.innerWidth - MIN_CONTENT_WIDTH);
    applyWidth(Math.round(Math.max(MIN_WIDTH,
      Math.min(maximum, drag.startWidth + event.clientX - drag.startX))));
  }

  return (
    <div
      ref={handleRef}
      className={styles.sidebarResizeHandle}
      role="separator"
      aria-orientation="vertical"
      aria-controls="app-sidebar"
      aria-label={t("controls.resizeNavigation")}
      title={t("controls.resizeNavigation")}
      onPointerDown={startDrag}
      onPointerMove={moveDrag}
      onPointerUp={finishDrag}
      onPointerCancel={finishDrag}
      onLostPointerCapture={finishDrag}
      onDoubleClick={() => {
        applyWidth(DEFAULT_WIDTH);
        saveWidth(DEFAULT_WIDTH);
      }}
    />
  );
}
