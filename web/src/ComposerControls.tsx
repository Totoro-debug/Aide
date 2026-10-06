import { Check, ChevronDown, FolderPen, Shield, ShieldCheck } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { KeyboardEvent, ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { ToolPermissionLevel } from "./protocol";
import styles from "./App.module.css";

const PERMISSION_LEVELS: ToolPermissionLevel[] = ["read-only", "workspace-write", "full-access"];
const PERMISSION_ICONS = { "read-only": ShieldCheck, "workspace-write": FolderPen, "full-access": Shield };

interface ComposerControlsProps {
  permission: ToolPermissionLevel;
  permissionDisabled: boolean;
  onPermissionChange: (permission: ToolPermissionLevel) => void;
  modelSummary: string;
  effortSummary: string;
  disabled: boolean;
  children: ReactNode;
}

export default function ComposerControls({
  permission, permissionDisabled, onPermissionChange, modelSummary, effortSummary, disabled, children,
}: ComposerControlsProps) {
  const { t } = useTranslation();
  const [open, setOpen] = useState<"permission" | "model" | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const permissionTriggerRef = useRef<HTMLButtonElement>(null);
  const modelTriggerRef = useRef<HTMLButtonElement>(null);
  const permissionMenuRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDivElement>(null);
  const focusLastRef = useRef(false);
  const PermissionIcon = PERMISSION_ICONS[permission];

  function closeMenu(restoreFocus = false) {
    setOpen(null);
    if (restoreFocus) {
      (open === "permission" ? permissionTriggerRef : modelTriggerRef).current?.focus();
    }
  }

  useLayoutEffect(() => {
    const menu = open === "permission" ? permissionMenuRef.current : open === "model" ? modelMenuRef.current : null;
    const controls = menu?.querySelectorAll<HTMLElement>("button:not(:disabled), select:not(:disabled)");
    if (controls?.length) controls[focusLastRef.current ? controls.length - 1 : 0].focus();
    focusLastRef.current = false;
  }, [open]);

  useEffect(() => {
    if (open === null) return;
    const dismissOutside = (event: Event) => {
      if (event.target instanceof Node && !rootRef.current?.contains(event.target)) setOpen(null);
    };
    document.addEventListener("pointerdown", dismissOutside);
    return () => {
      document.removeEventListener("pointerdown", dismissOutside);
    };
  }, [open]);

  function handleTriggerKey(event: KeyboardEvent<HTMLButtonElement>, menu: "permission" | "model") {
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    event.preventDefault();
    focusLastRef.current = event.key === "ArrowUp";
    setOpen(menu);
  }

  function handlePermissionKey(event: KeyboardEvent<HTMLDivElement>) {
    const buttons = [...(permissionMenuRef.current?.querySelectorAll<HTMLButtonElement>("button:not(:disabled)") ?? [])];
    const index = buttons.indexOf(document.activeElement as HTMLButtonElement);
    let next = index;
    if (event.key === "ArrowDown") next = (index + 1) % buttons.length;
    else if (event.key === "ArrowUp") next = (index + buttons.length - 1) % buttons.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = buttons.length - 1;
    else return;
    event.preventDefault();
    buttons[next]?.focus();
  }

  return (
    <div className={styles.composerControls} ref={rootRef} onBlur={(event) => {
      if (!event.currentTarget.contains(event.relatedTarget)) setOpen(null);
    }} onKeyDown={(event) => {
      if (event.key === "Escape" && open !== null) {
        event.preventDefault();
        event.stopPropagation();
        closeMenu(true);
      }
    }}>
      <div className={styles.composerPermissionControl}>
        <button
          ref={permissionTriggerRef}
          id="composer-permission-trigger"
          className={styles.composerControlTrigger}
          type="button"
          aria-label={t("conversation.clientPermission")}
          aria-haspopup="menu"
          aria-expanded={open === "permission"}
          aria-controls="composer-permission-menu"
          data-value={permission}
          disabled={permissionDisabled || disabled}
          onClick={() => setOpen(open === "permission" ? null : "permission")}
          onKeyDown={(event) => handleTriggerKey(event, "permission")}
        >
          <PermissionIcon size={14} aria-hidden="true" />
          <span>{t(`conversation.permissionLevels.${permission}`)}</span>
          <ChevronDown className={styles.composerChevron} size={12} aria-hidden="true" />
        </button>
        <div
          ref={permissionMenuRef}
          id="composer-permission-menu"
          className={`${styles.composerMenu} ${styles.composerPermissionMenu}`}
          role="menu"
          aria-label={t("conversation.clientPermission")}
          hidden={open !== "permission"}
          onKeyDown={handlePermissionKey}
        >
          {PERMISSION_LEVELS.map((level) => {
            const Icon = PERMISSION_ICONS[level];
            return (
              <button
                key={level}
                className={styles.composerMenuOption}
                type="button"
                role="menuitemradio"
                tabIndex={-1}
                aria-checked={permission === level}
                data-value={level}
                disabled={permissionDisabled || disabled}
                onClick={() => {
                  closeMenu(true);
                  onPermissionChange(level);
                }}
              >
                <Icon size={14} aria-hidden="true" />
                <span>{t(`conversation.permissionLevels.${level}`)}</span>
                {permission === level ? <Check size={14} aria-hidden="true" /> : null}
              </button>
            );
          })}
        </div>
      </div>
      <div className={styles.composerModelControl}>
        <button
          ref={modelTriggerRef}
          id="composer-model-trigger"
          className={styles.composerControlTrigger}
          type="button"
          aria-label={t("conversation.modelAndEffort")}
          title={`${modelSummary} · ${effortSummary}`}
          aria-expanded={open === "model"}
          aria-controls="composer-model-menu"
          disabled={disabled}
          onClick={() => setOpen(open === "model" ? null : "model")}
          onKeyDown={(event) => handleTriggerKey(event, "model")}
        >
          <span className={styles.composerModelName}>{modelSummary}</span>
          <span lang="en">{effortSummary}</span>
          <ChevronDown className={styles.composerChevron} size={12} aria-hidden="true" />
        </button>
        <div
          ref={modelMenuRef}
          id="composer-model-menu"
          className={`${styles.composerMenu} ${styles.composerModelMenu}`}
          hidden={open !== "model"}
          onChange={() => closeMenu(true)}
        >
          {children}
        </div>
      </div>
    </div>
  );
}
