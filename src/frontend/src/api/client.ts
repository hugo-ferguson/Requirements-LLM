const BASE_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  // The browser has to set its own multipart Content-Type (it carries the
  // boundary), so only JSON bodies get the header.
  const isFormData = init?.body instanceof FormData;
  const headers = isFormData
    ? init?.headers
    : { "Content-Type": "application/json", ...init?.headers };

  const res = await fetch(`${BASE_URL}${path}`, { ...init, headers });

  if (!res.ok) {
    throw new ApiError(res.status, await res.text());
  }

  if (res.status === 204) {
    return undefined as T;
  }

  return (await res.json()) as T;
}
