from pathlib import Path

# Folders to ignore
IGNORED_DIRS = {
    ".git",
    ".ruff_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
}

# Name of this script, so it can also be excluded from the tree
SCRIPT_NAME = Path(__file__).name


def print_tree(directory: Path, prefix: str = ""):
    """Print the directory structure as a tree."""

    try:
        entries = sorted(
            [
                entry
                for entry in directory.iterdir()
                if not (entry.is_dir() and entry.name in IGNORED_DIRS)
                and entry.name != SCRIPT_NAME
            ],
            key=lambda x: (not x.is_dir(), x.name.lower()),
        )
    except PermissionError:
        print(f"{prefix}└── [Permission Denied]")
        return

    for index, entry in enumerate(entries):
        is_last = index == len(entries) - 1

        connector = "└── " if is_last else "├── "
        print(f"{prefix}{connector}{entry.name}")

        if entry.is_dir():
            extension = "    " if is_last else "│   "
            print_tree(entry, prefix + extension)


if __name__ == "__main__":
    # Directory where this Python script is located
    root = Path(__file__).resolve().parent

    print(root.name)
    print_tree(root)
