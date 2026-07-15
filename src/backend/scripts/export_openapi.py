"""Writes the FastAPI app's OpenAPI schema to src/openapi.json.

Run this after changing routes/models, then run `npm run generate-api`
in frontend/ to refresh the generated TypeScript types.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


def main() -> None:
    schema = app.openapi()
    out_path = Path(__file__).resolve().parents[2] / "openapi.json"
    out_path.write_text(json.dumps(schema, indent=2))
    print(f"Wrote OpenAPI schema to {out_path}")


if __name__ == "__main__":
    main()
