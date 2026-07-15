import type { Item } from "../api/items";

interface ItemListProps {
  items: Item[];
}

export function ItemList({ items }: ItemListProps) {
  if (items.length === 0) {
    return <p>No items yet.</p>;
  }

  return (
    <ul>
      {items.map((item) => (
        <li key={item.id}>
          {item.name}{" "}
          <small>({new Date(item.created_at).toLocaleString()})</small>
        </li>
      ))}
    </ul>
  );
}
