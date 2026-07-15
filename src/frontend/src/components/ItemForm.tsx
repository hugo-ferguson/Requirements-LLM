import { useState } from "react";
import type { FormEvent } from "react";

interface ItemFormProps {
  onSubmit: (name: string) => Promise<void>;
}

export function ItemForm({ onSubmit }: ItemFormProps) {
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!name.trim()) return;

    setSubmitting(true);
    try {
      await onSubmit(name.trim());
      setName("");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="text"
        value={name}
        onChange={(event) => setName(event.target.value)}
        placeholder="New item name"
        disabled={submitting}
      />
      <button type="submit" disabled={submitting || !name.trim()}>
        Add item
      </button>
    </form>
  );
}
