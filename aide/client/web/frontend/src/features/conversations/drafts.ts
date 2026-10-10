import { createContext, useContext } from "react";

interface ConversationDrafts {
  drafts: { current: Record<string, Record<string, string>> };
  version: number;
  changed: () => void;
}

export const ConversationDraftsContext = createContext<ConversationDrafts | null>(null);

export function useConversationDrafts(): ConversationDrafts {
  const drafts = useContext(ConversationDraftsContext);
  if (drafts === null) throw new Error("Conversation drafts require their provider.");
  return drafts;
}
