#!/usr/bin/env python3
"""
Check if classes referenced in API documentation have proper docstrings.

Usage:
    python playground/scripts/check_docs_api.py
    python playground/scripts/check_docs_api.py --check-methods
    python playground/scripts/check_docs_api.py --warn-only
"""

import sys
from pathlib import Path

from xdeploy.common.docs_utils import (
    BLUE,
    GREEN,
    RED,
    RESET,
    YELLOW,
    check_docstring,
    check_method_docstrings,
    extract_api_references,
    get_class_from_path,
)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Check if classes in API documentation have docstrings"
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "docs",
        help="Path to documentation directory",
    )
    parser.add_argument(
        "--check-methods",
        action="store_true",
        help="Also check method docstrings",
    )
    parser.add_argument(
        "--warn-only",
        action="store_true",
        help="Only warn, don't exit (for CI)",
    )

    args = parser.parse_args()

    # Switch to project root directory
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    print(f"{BLUE}Checking API documentation...{RESET}")
    print(f"Documentation directory: {args.docs_dir}")
    print(f"Python: {sys.executable}\n")

    # Extract all API references
    references = extract_api_references(args.docs_dir)
    print(f"{BLUE}Found {len(references)} API references{RESET}\n")

    # Check each reference
    errors = []
    warnings = []

    for file_path, class_path in references:
        print(f"{BLUE}Checking: {class_path}{RESET} (in {file_path})")

        cls = get_class_from_path(class_path)
        has_doc, msg = check_docstring(cls)

        if not has_doc:
            error_msg = f"{file_path}: {class_path} - {msg}"
            if cls is None:
                errors.append(error_msg)
                print(f"  {RED}✗ Error: {msg}{RESET}")
            else:
                warnings.append(error_msg)
                print(f"  {YELLOW}⚠ Warning: {msg}{RESET}")
        else:
            print(f"  {GREEN}✓ {msg}{RESET}")

        # Check method docstrings
        if args.check_methods and cls is not None:
            method_results = check_method_docstrings(cls)
            missing_methods = [
                (name, msg)
                for name, has_doc, msg in method_results
                if not has_doc
            ]
            if missing_methods:
                print(f"  {YELLOW}  Methods missing docstrings:{RESET}")
                for name, msg in missing_methods[:5]:  # Only show first 5
                    print(f"    {YELLOW}- {name}: {msg}{RESET}")
                if len(missing_methods) > 5:
                    print(
                        f"    {YELLOW}... and {len(missing_methods) - 5} more methods{RESET}"
                    )

        print()

    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}Check results:{RESET}")
    print(f"  Total: {len(references)}")
    print(
        f"  {GREEN}Passed: {len(references) - len(errors) - len(warnings)}{RESET}"
    )
    if warnings:
        print(f"  {YELLOW}Warnings: {len(warnings)}{RESET}")
    if errors:
        print(f"  {RED}Errors: {len(errors)}{RESET}")

    if errors:
        print(f"\n{RED}Error details:{RESET}")
        for error in errors:
            print(f"  {RED}✗ {error}{RESET}")

    if warnings:
        print(f"\n{YELLOW}Warning details:{RESET}")
        for warning in warnings[:10]:  # Only show first 10 warnings
            print(f"  {YELLOW}⚠ {warning}{RESET}")
        if len(warnings) > 10:
            print(
                f"  {YELLOW}... and {len(warnings) - 10} more warnings{RESET}"
            )

    # Exit code
    if errors:
        if args.warn_only:
            print(
                f"\n{YELLOW}Warning mode: errors found but continuing{RESET}"
            )
            return 0
        else:
            print(f"\n{RED}Check failed! Please fix the errors above.{RESET}")
            return 1
    elif warnings:
        print(
            f"\n{YELLOW}Check completed with warnings. Consider adding docstrings.{RESET}"
        )
        return 0
    else:
        print(f"\n{GREEN}All checks passed!{RESET}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
