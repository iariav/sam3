import sys
import shutil
import re
from pathlib import Path


NUMBERED_JSON_REGEX = re.compile(r'^(\d+)\.json$')


def get_max_number_in_folder(folder: Path) -> int:
    """
    Return the highest numeric JSON filename in the folder (e.g., 10 for '10.json').
    If none are found, return 0.
    """
    max_num = 0
    for f in folder.iterdir():
        if f.is_file():
            m = NUMBERED_JSON_REGEX.match(f.name)
            if m:
                n = int(m.group(1))
                if n > max_num:
                    max_num = n
    return max_num


def copy_and_renumber_json_and_png(src_folder: Path, dst_folder: Path) -> None:
    """
    Copy numbered JSON files from src_folder to dst_folder.
    For each JSON, also copy a PNG with the same base name if it exists.

    Files are renamed to continue the numbering in dst_folder.
    Example:
      dst has: 1.json, 2.json
      src has: 1.json, 1.png, 2.json, 2.png
      result: 3.json, 3.png, 4.json, 4.png in dst
    """
    # Ensure destination exists
    dst_folder.mkdir(parents=True, exist_ok=True)

    # Find the current max number in destination
    current_max = get_max_number_in_folder(dst_folder)

    # Collect and sort numeric JSON files from source
    numbered_files = []
    for f in src_folder.iterdir():
        if f.is_file():
            m = NUMBERED_JSON_REGEX.match(f.name)
            if m:
                n = int(m.group(1))
                numbered_files.append((n, f))

    # Sort by their original numeric name
    numbered_files.sort(key=lambda x: x[0])

    # Copy JSON + PNG with new incremented names
    next_number = current_max
    for num, src_json in numbered_files:
        next_number += 1
        new_base = f"{next_number}"

        # Copy JSON
        new_json_name = f"{new_base}.json"
        dst_json = dst_folder / new_json_name
        shutil.copy2(src_json, dst_json)
        print(f"Copied {src_json} -> {dst_json}")

        # Copy PNG with same original number, if it exists
        src_png = src_folder / f"{num}.png"
        if src_png.exists() and src_png.is_file():
            new_png_name = f"{new_base}.png"
            dst_png = dst_folder / new_png_name
            shutil.copy2(src_png, dst_png)
            print(f"Copied {src_png} -> {dst_png}")
        else:
            print(f"Warning: PNG for {src_json.name} not found ({src_png.name}), skipping image.")


def main():
    if len(sys.argv) != 3:
        print("Usage: python copy_numbered_json.py <source_folder> <destination_folder>")
        sys.exit(1)

    src = Path(sys.argv[1]).expanduser().resolve()
    dst = Path(sys.argv[2]).expanduser().resolve()

    if not src.is_dir():
        print(f"Source folder does not exist or is not a directory: {src}")
        sys.exit(1)

    copy_and_renumber_json_and_png(src, dst)


if __name__ == "__main__":
    main()
