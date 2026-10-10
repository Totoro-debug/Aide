import type { ReactNode } from "react";
import { useCallback, useMemo, useRef, useState } from "react";

import { ConversationDraftsContext } from "./drafts";

export function ConversationDraftsProvider({ children }: { children: ReactNode }) {
  const drafts = useRef<Record<string, Record<string, string>>>({});
  const [version, setVersion] = useState(0);
  const changed = useCallback(() => setVersion((current) => current + 1), []);
  const value = useMemo(() => ({ drafts, version, changed }), [version, changed]);
  return <ConversationDraftsContext.Provider value={value}>{children}</ConversationDraftsContext.Provider>;
}
