import { useEffect, useState } from "react";
import { itemsApi } from "./api/items";
import type { Item } from "./api/items";
import { ItemForm } from "./components/ItemForm";
import { ItemList } from "./components/ItemList";

export default function App() {
  const [items, setItems] = useState<Item[]>([]);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      setItems(await itemsApi.list());
      setError(null);
    } catch {
      setError("Failed to load items. Is the backend running?");
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleCreate(name: string) {
    await itemsApi.create({ name });
    await refresh();
  }

  return (
    <main style={{ maxWidth: 480, margin: "2rem auto", fontFamily: "sans-serif" }}>
      <h1>Items</h1>
      {error && <p style={{ color: "red" }}>{error}</p>}
      <ItemForm onSubmit={handleCreate} />
      <ItemList items={items} />
    </main>
  );
}
