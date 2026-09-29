import i18n from "i18next";
import { initReactI18next } from "react-i18next";

const resources = {
  en: {
    translation: {
      app: { name: "MyClaw", subtitle: "Local workbench" },
      nav: { status: "Status", projects: "Projects" },
      controls: {
        language: "Language",
        theme: "Theme",
        system: "System",
        light: "Light",
        dark: "Dark",
        details: "Connection details",
        close: "Close",
        addProject: "Add project",
        refresh: "Refresh projects",
        retry: "Retry",
        cancel: "Cancel",
        register: "Register project",
        registering: "Registering",
        resumeSchedule: "Resume schedule",
        resuming: "Resuming",
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
      projects: {
        title: "Projects",
        description: "Registered local directories available to the Web Interface.",
        addTitle: "Register a local directory",
        pathLabel: "Absolute local path",
        pathPlaceholder: "C:\\Users\\you\\Projects\\example",
        pathHint: "Use an existing directory path on this computer.",
        emptyTitle: "No projects registered",
        emptyDescription: "Register an existing directory to make it available here.",
        loading: "Loading projects",
        loadError: "Projects could not be loaded.",
        unavailable: "Unavailable",
        available: "Available",
        scheduleReady: "Schedule active",
        schedulePaused: "Schedule paused for review",
        scheduleUnavailable: "Schedule unavailable",
        scheduleState: {
          available: "Schedule active",
          unavailable: "Schedule unavailable",
          awaiting_resume: "Schedule paused for review",
          removing: "Project removal in progress",
          failed: "Schedule unavailable",
        },
        reviewTitle: "Review Schedule Jobs",
        reviewDescription: "Resume saved Jobs for {{name}}? Due Jobs may run immediately.",
        reviewStatus: {
          overdue: "Overdue; may run when resumed",
          upcoming: "Upcoming",
          next_on_resume: "Next matching time after resume",
          completed: "Already completed",
        },
        everySchedule: "Every {{seconds}} seconds",
        cronSchedule: "Cron {{expression}} ({{timezone}})",
        atSchedule: "At {{time}}",
        pathAbsoluteError: "Enter an absolute local directory path.",
        pathOverlapError: "The directory must not overlap Agent Home.",
        pathMissingError: "Enter an existing directory path.",
        pathInvalidError: "Enter a usable local directory path.",
        staleReviewError: "Saved Jobs changed. Review the list and try again.",
        persistenceError: "Project records could not be read or saved safely.",
        savedJobs: "Saved Schedule Jobs",
        noSavedJobs: "No saved Schedule Jobs",
        registeredNotice: "Project registered.",
        registeredPausedNotice: "Project registered. Saved Schedule Jobs remain paused until resumed.",
        actionError: "The Project action could not be completed.",
        authenticationRequired: "Authenticate with the local service to manage Projects.",
      },
      footer: { localOnly: "Loopback service" },
    },
  },
  "zh-CN": {
    translation: {
      app: { name: "MyClaw", subtitle: "本地工作台" },
      nav: { status: "状态", projects: "项目" },
      controls: {
        language: "语言",
        theme: "主题",
        system: "跟随系统",
        light: "浅色",
        dark: "深色",
        details: "连接详情",
        close: "关闭",
        addProject: "登记项目",
        refresh: "刷新项目",
        retry: "重试",
        cancel: "取消",
        register: "登记项目",
        registering: "登记中",
        resumeSchedule: "恢复调度",
        resuming: "恢复中",
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
      projects: {
        title: "项目",
        description: "可在 Web 界面使用的已登记本地目录。",
        addTitle: "登记本地目录",
        pathLabel: "本地绝对路径",
        pathPlaceholder: "D:\\Projects\\example",
        pathHint: "填写此电脑上已经存在的目录路径。",
        emptyTitle: "尚未登记项目",
        emptyDescription: "登记一个已有目录后，它会显示在这里。",
        loading: "正在加载项目",
        loadError: "项目加载失败。",
        unavailable: "不可用",
        available: "可用",
        scheduleReady: "调度已启用",
        schedulePaused: "调度已暂停，等待确认",
        scheduleUnavailable: "调度不可用",
        scheduleState: {
          available: "调度已启用",
          unavailable: "调度不可用",
          awaiting_resume: "调度已暂停，等待确认",
          removing: "正在移除项目",
          failed: "调度不可用",
        },
        reviewTitle: "检查定时任务",
        reviewDescription: "恢复 {{name}} 的已保存任务？到期任务可能立即执行。",
        reviewStatus: {
          overdue: "已到期，恢复后可能立即执行",
          upcoming: "尚未到期",
          next_on_resume: "恢复后在下次匹配时间执行",
          completed: "已经完成",
        },
        everySchedule: "每 {{seconds}} 秒",
        cronSchedule: "定时表达式 {{expression}}（{{timezone}}）",
        atSchedule: "指定时间 {{time}}",
        pathAbsoluteError: "请输入本地目录的绝对路径。",
        pathOverlapError: "目录不能与 Agent Home 重叠。",
        pathMissingError: "请输入已存在的目录路径。",
        pathInvalidError: "请输入可用的本地目录路径。",
        staleReviewError: "已保存任务发生变化，请重新检查后再试。",
        persistenceError: "无法安全读取或保存项目记录。",
        savedJobs: "已保存的 Schedule Job",
        noSavedJobs: "没有已保存的 Schedule Job",
        registeredNotice: "项目已登记。",
        registeredPausedNotice: "项目已登记，已保存的 Schedule Job 在恢复前保持暂停。",
        actionError: "项目操作未完成。",
        authenticationRequired: "请先连接本地服务再管理项目。",
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
