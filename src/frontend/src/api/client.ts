const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

/** A non-2xx response. `message` is the raw response body. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

/** Fetches JSON from the backend. Throws {@link ApiError} on non-2xx; 204 gives `undefined`. */
export async function request<ResponseBody>(
  path: string,
  options?: RequestInit,
): Promise<ResponseBody> {
  // The browser sets the multipart Content-Type itself, as it includes the boundary.
  const isFormData = options?.body instanceof FormData;
  const headers = isFormData
    ? options?.headers
    : { "Content-Type": "application/json", ...options?.headers };

  const response = await fetch(`${BASE_URL}${path}`, { ...options, headers });

  if (!response.ok) {
    throw new ApiError(response.status, await response.text());
  }

  if (response.status === 204) {
    return undefined as ResponseBody;
  }

  return (await response.json()) as ResponseBody;
}
