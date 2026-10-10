
export interface SettingsReturnLocation {
  pathname: string;
  search: string;
  hash: string;
  scrollTop: number;
  conversationScrollTop?: number;
}

export interface SettingsNavigationState {
  returnTo?: SettingsReturnLocation;
}
