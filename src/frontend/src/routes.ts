export const ROUTES = {
  root: "/",
  input: (sessionId: string | number) => `/sessions/${sessionId}/input`,
  acReview: (sessionId: string | number) => `/sessions/${sessionId}/ac-review`,
  uatReview: (sessionId: string | number) => `/sessions/${sessionId}/uat-review`,
  export: (sessionId: string | number) => `/sessions/${sessionId}/export`,
} as const;

// Relative child-path patterns used in route declarations, nested under "/sessions/:sessionId".
export const TAB_PATHS = {
  input: "input",
  acReview: "ac-review",
  uatReview: "uat-review",
  export: "export",
} as const;
