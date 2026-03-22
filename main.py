import argparse
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path


LVIS_VAL_ZIP_URL = "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip"
LVIS_VAL_JSON_NAME = "lvis_v1_val.json"


def download_lvis_categories(output_path: Path) -> None:
    """Download LVIS annotations and save full categories dictionary."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_path = Path(tmp_dir) / "lvis_v1_val.json.zip"

        print(f"Downloading: {LVIS_VAL_ZIP_URL}")
        urllib.request.urlretrieve(LVIS_VAL_ZIP_URL, zip_path)

        with zipfile.ZipFile(zip_path, "r") as archive:
            with archive.open(LVIS_VAL_JSON_NAME) as annotations_file:
                annotations = json.load(annotations_file)

    categories = annotations.get("categories", [])
    categories_dict = {
        str(category["id"]): {
            "name": category["name"],
            "synonyms": category.get("synonyms", []),
        }
        for category in categories
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(categories_dict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(categories_dict)} categories to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download full LVIS categories dictionary."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("lvis_categories_full.json"),
        help="Path to output JSON file (default: lvis_categories_full.json).",
    )
    args = parser.parse_args()

    download_lvis_categories(args.output)


if __name__ == "__main__":
    main()
