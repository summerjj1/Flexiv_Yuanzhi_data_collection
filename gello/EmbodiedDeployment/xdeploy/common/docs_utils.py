"""Utilities for checking API documentation docstrings."""

import importlib
import re
from pathlib import Path
from typing import List, Tuple

# Color output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"


def extract_api_references(docs_dir: Path) -> List[Tuple[str, str]]:
    """Extract API references from documentation files.

    Args:
        docs_dir: Path to documentation directory

    Returns:
        List of (file_path, class_path) tuples
    """
    references = []
    pattern = re.compile(r"^::: (.+)$", re.MULTILINE)

    # Exclude example documentation
    exclude_dirs = {
        "cursor",
        "getting-started",
        "tutorials",
        "examples",
        "architecture",
    }
    exclude_files = {"DOCUMENTATION_GUIDE.md", "README.md", "API_FIXES.md"}

    for md_file in docs_dir.rglob("*.md"):
        # Skip excluded directories and files
        relative_path = md_file.relative_to(docs_dir)
        if any(part in exclude_dirs for part in relative_path.parts):
            continue
        if md_file.name in exclude_files:
            continue

        content = md_file.read_text(encoding="utf-8")
        matches = pattern.findall(content)
        for match in matches:
            # Remove possible option lines
            class_path = match.strip().split()[0] if match.strip() else ""
            if class_path and class_path.startswith("xdeploy."):
                # Skip references in example code
                if "xdeploy.module.Class" in class_path:
                    continue
                references.append((str(relative_path), class_path))

    return references


def get_class_from_path(class_path: str):
    """Import and return class object or module from class path string.

    Args:
        class_path: Full path to the class, e.g., "xdeploy.robot.robot.Robot"
                    or module path "xdeploy.common.logger_utils"

    Returns:
        Class object or module object, returns None if import fails
    """
    try:
        parts = class_path.split(".")

        # Try importing as a module
        try:
            module = importlib.import_module(class_path)
            return module
        except (ImportError, AttributeError):
            pass

        # Try importing as a class
        if len(parts) > 1:
            module_path = ".".join(parts[:-1])
            class_name = parts[-1]
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name, None)
            if cls is not None:
                return cls

        return None
    except Exception as e:
        return None


def check_docstring(obj) -> Tuple[bool, str]:
    """Check if object has a docstring.

    Args:
        obj: Object to check (class, function, or module)

    Returns:
        (has_docstring, message) tuple
    """
    if obj is None:
        return False, "Class/module does not exist"

    # Check if it's a module
    import types

    if isinstance(obj, types.ModuleType):
        doc = getattr(obj, "__doc__", None)
        if doc is None or not doc.strip():
            return False, "Module missing docstring"
        if len(doc.strip()) < 10:
            return (
                False,
                f"Module docstring too short ({len(doc.strip())} characters)",
            )
        return True, "✓"

    # Check class or function
    doc = getattr(obj, "__doc__", None)
    if doc is None:
        return False, "Missing docstring"
    if not doc.strip():
        return False, "Docstring is empty"
    if len(doc.strip()) < 10:
        return False, f"Docstring too short ({len(doc.strip())} characters)"

    return True, "✓"


def check_method_docstrings(cls) -> List[Tuple[str, bool, str]]:
    """Check if public methods of a class have docstrings.

    Args:
        cls: Class to check

    Returns:
        List of (method_name, has_docstring, message) tuples
    """
    results = []
    if cls is None:
        return results

    for name in dir(cls):
        if name.startswith("_"):
            continue

        attr = getattr(cls, name)
        if not callable(attr):
            continue

        # Skip special methods (except __init__)
        if name.startswith("__") and name != "__init__":
            continue

        # Check if it's a method (not a function)
        if not hasattr(attr, "__self__"):
            continue

        has_doc, msg = check_docstring(attr)
        results.append((name, has_doc, msg))

    return results
