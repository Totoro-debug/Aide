import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      app: { name: "MyClaw", subtitle: "Local workbench" },
      nav: { status: "Status" },
      controls: {
        language: "Language",
        theme: "Theme",
        system: "System",
        light: "Light",
        dark: "Dark",
        details: "Connection details",
        close: "Close",
      },
      status: {
        title: "Service status",
        ready: "Ready",
        starting: "Starting",
        reconnecting: "Reconnecting",
        draining: "Draining",
        stopped: "Stopped",
        online: "Online",
        offline: "Offline",
        checking: "Checking",
        workspaces: "Active workspaces",
        protocol: "Protocol",
        instance: "Service instance",
        connection: "Connection",
        authenticationRequired: "Open this workbench from the local MyClaw command.",
        unavailable: "The local service is unavailable.",
        detailsTitle: "Connection details",
        detailsDescription: "Current service identity and transport state.",
      },
      footer: { localOnly: "Loopback service" },
    },
  },
  "zh-CN": {
    translation: {
      app: { name: "MyClaw", subtitle: "本地工作台" },
      nav: { status: "状态" },
      controls: {
        language: "语言",
        theme: "主题",
        system: "跟随系统",
        light: "浅色",
        dark: "深色",
        details: "连接详情",
        close: "关闭",
      },
      status: {
        title: "服务状态",
        ready: "就绪",
        starting: "启动中",
        reconnecting: "恢复连接中",
        draining: "收尾中",
        stopped: "已停止",
        online: "在线",
        offline: "离线",
        checking: "检查中",
        workspaces: "活动工作区",
        protocol: "协议",
        instance: "服务实例",
        connection: "连接",
        authenticationRequired: "请从本地 MyClaw 命令打开此工作台。",
        unavailable: "本地服务不可用。",
        detailsTitle: "连接详情",
        detailsDescription: "当前服务身份和传输状态。",
      },
      footer: { localOnly: "仅限本机服务" },
    },
  },
} as const;

const language = readPreference("myclaw.language") ?? detectLanguage();

void i18n.use(initReactI18next).init({
  resources,
  lng: language,
  fallbackLng: "en",
  interpolation: { escapeValue: false },
});

i18n.on("languageChanged", (value) => {
  writePreference("myclaw.language", value);
  document.documentElement.lang = value;
});

document.documentElement.lang = language;

export default i18n;

function detectLanguage(): "en" | "zh-CN" {
  return navigator.language.toLowerCase().startsWith("zh") ? "zh-CN" : "en";
}

function readPreference(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    return null;
  }
}

function writePreference(key: string, value: string): void {
  try {
    window.localStorage.setItem(key, value);
  } catch {
    // Preferences are optional when storage is unavailable.
  }
}
