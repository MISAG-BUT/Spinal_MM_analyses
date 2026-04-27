#!/usr/bin/env python3
"""
Script for renaming identifiers in JSON files
inside all subfolders of selected directories.

Base folder:
    /home/nohel/DATA/MultipleMyeloma_analyses/full_models

Target folders:
    longi_summary_all
    longi_summary_larger_than_0_3_cubic_cm
    longi_summary_larger_than_0_5_cubic_cm

The script scans all nested subfolders and processes all JSON files.
Uses temporary placeholders to avoid chained replacements.
"""

import os
import sys

#base_dir = "/home/nohel/DATA/MultipleMyeloma_analyses/full_models"
base_dir = "/home/nohel/DATA/MultipleMyeloma_analyses/zero_input_models"
target_dirs = [
    "longi_summary_all",
    "longi_summary_larger_than_0_3_cubic_cm",
    "longi_summary_larger_than_0_5_cubic_cm",
]

# Replacement mapping
replacements = {
    "Myel_069": "Myel_012_b",
    "Myel_012": "Myel_012_a",
    "Myel_047": "Myel_018_b",
    "Myel_018": "Myel_018_a",
    "Myel_024": "Myel_023_b",
    "Myel_023": "Myel_023_a",
    "Myel_059": "Myel_043_b",
    "Myel_043": "Myel_043_a",
    "Myel_070": "Myel_052_b",
    "Myel_052": "Myel_052_a",
}


def replace_content(content):
    """Apply safe replacements using temporary placeholders."""
    placeholders = {}

    # Step 1: old names -> temporary tokens
    for i, old in enumerate(replacements):
        token = f"__TMP_{i}__"
        placeholders[token] = replacements[old]
        content = content.replace(old, token)

    # Step 2: tokens -> final names
    for token, new in placeholders.items():
        content = content.replace(token, new)

    return content


def process_file(filepath):
    """Process one JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    content = replace_content(content)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    processed = 0

    for folder in target_dirs:
        root_dir = os.path.join(base_dir, folder)

        if not os.path.exists(root_dir):
            print(f"Skipping missing folder: {root_dir}")
            continue

        print(f"\nScanning: {root_dir}")

        for current_path, _, files in os.walk(root_dir):
            for filename in files:
                if filename.endswith(".json"):
                    filepath = os.path.join(current_path, filename)

                    try:
                        process_file(filepath)
                        print(f"Processed: {filepath}")
                        processed += 1

                    except Exception as e:
                        print(f"Error: {filepath}")
                        print(e)

    print(f"\nDone. Processed {processed} JSON files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())