/** Tab paths relative to "/sessions/:sessionId". */
export const TAB_PATHS = {
  input: "input",
  acReview: "ac-review",
  uatReview: "uat-review",
  export: "export",
} as const;

/** Absolute paths to navigate to. */
export const ROUTES = {
  root: "/",
  input: (sessionId: string | number) => `/sessions/${sessionId}/${TAB_PATHS.input}`,
  acReview: (sessionId: string | number) => `/sessions/${sessionId}/${TAB_PATHS.acReview}`,
  uatReview: (sessionId: string | number) => `/sessions/${sessionId}/${TAB_PATHS.uatReview}`,
  export: (sessionId: string | number) => `/sessions/${sessionId}/${TAB_PATHS.export}`,
} as const;
