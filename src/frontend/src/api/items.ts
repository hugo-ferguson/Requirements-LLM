import { request } from "./client";
import type { paths } from "./schema";

export type Item =
  paths["/items"]["get"]["responses"]["200"]["content"]["application/json"][number];
export type ItemCreate =
  paths["/items"]["post"]["requestBody"]["content"]["application/json"];

export const itemsApi = {
  list: (): Promise<Item[]> => request<Item[]>("/items"),
  create: (data: ItemCreate): Promise<Item> =>
    request<Item>("/items", { method: "POST", body: JSON.stringify(data) }),
};
