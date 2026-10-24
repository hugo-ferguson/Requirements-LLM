"""Write the app's OpenAPI schema to src/openapi.json.

Then run `npm run generate-api` in src/frontend to refresh the types.
"""

import json
import sys
from pathlib import Path

# Lets the script import `app` when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import app  # noqa: E402


def main() -> None:
    """Export the schema and print where it was written."""
    schema = app.openapi()
    output_path = Path(__file__).resolve().parents[2] / "openapi.json"
    output_path.write_text(json.dumps(schema, indent=2))
    print(f"Wrote OpenAPI schema to {output_path}")


if __name__ == "__main__":
    main()
