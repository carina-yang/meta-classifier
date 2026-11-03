#!/usr/bin/env python3
"""
Interactive disagreement review tool.

Usage:
    python scripts/review/review_disagreements.py [--category <name>] [--batch-size <n>]

Categories:
    - high_confidence_model_likely_right (40 items)
    - high_confidence_model_likely_wrong (16 items)
    - ambiguous_needs_context (29 items)
    - clear_cut_cases (112 items)
"""

import csv
import json
import sys
import os
from pathlib import Path

# Navigate up to project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)

# Add src to path
sys.path.insert(0, str(project_root / 'src'))
from taxonomy import TAXONOMY, LABELS

def display_record(idx, row, total):
    """Display a single disagreement record."""
    print("\n" + "=" * 80)
    print(f"RECORD {idx + 1} of {total}")
    print("=" * 80)
    print(f"Table:  {row['table_name']}")
    print(f"Column: {row['column_name']}")
    print(f"Type:   {row['data_type']}")
    print(f"Mode:   {row['mode']}")
    if row.get('source_class'):
        print(f"Source: {row['source_class']} (for hard negatives)")
    print()
    print(f"Intended class:  {row['intended_class']}")
    print(f"Model labeled:   {row['labeled_class']}")
    print(f"Model confidence: {row['confidence']}")
    print()

def show_taxonomy_help(label):
    """Show taxonomy definition for a label."""
    if label in TAXONOMY:
        spec = TAXONOMY[label]
        print(f"\n{label}:")
        print(f"  {spec['definition']}")
        print(f"  Examples: {', '.join(spec['include'][:5])}")

def review_interactive(records, category_name=""):
    """Interactive review of records."""
    if not records:
        print("No records to review!")
        return []

    print(f"\n{'=' * 80}")
    print(f"REVIEWING: {category_name}")
    print(f"Total records: {len(records)}")
    print(f"{'=' * 80}")
    print("\nCommands:")
    print("  1-10: Select class by number")
    print("  i: Use intended class")
    print("  m: Use model's labeled class")
    print("  ?: Show all class definitions")
    print("  s: Skip this record")
    print("  q: Quit and save progress")
    print()

    # Show class numbers
    print("Classes:")
    for i, label in enumerate(LABELS, 1):
        print(f"  {i}. {label}")
    print()

    results = []

    for idx, row in enumerate(records):
        display_record(idx, row, len(records))

        while True:
            cmd = input(f"Decision [{row['intended_class']}]: ").strip().lower()

            if cmd == 'q':
                print(f"\nSaving progress... Reviewed {len(results)} of {len(records)}")
                return results
            elif cmd == 's':
                break
            elif cmd == 'i':
                results.append({**row, 'human_label': row['intended_class'], 'review_decision': 'kept_intended'})
                print(f"✓ Kept intended: {row['intended_class']}")
                break
            elif cmd == 'm':
                results.append({**row, 'human_label': row['labeled_class'], 'review_decision': 'kept_model'})
                print(f"✓ Kept model: {row['labeled_class']}")
                break
            elif cmd == '?':
                for label in LABELS:
                    show_taxonomy_help(label)
            elif cmd.isdigit() and 1 <= int(cmd) <= len(LABELS):
                chosen = LABELS[int(cmd) - 1]
                results.append({**row, 'human_label': chosen, 'review_decision': 'manual_override'})
                print(f"✓ Manually set to: {chosen}")
                break
            else:
                print("Invalid command. Try again.")

    return results

def batch_review(records, category_name=""):
    """Batch review showing multiple records at once."""
    if not records:
        print("No records to review!")
        return []

    print(f"\n{'=' * 80}")
    print(f"BATCH REVIEW: {category_name}")
    print(f"{'=' * 80}\n")

    # Show all records
    for idx, row in enumerate(records):
        print(f"{idx:3d}. {row['table_name']:30s} {row['column_name']:30s}")
        print(f"      Intended: {row['intended_class']:25s} Model: {row['labeled_class']:25s} [{row['confidence']}]")
        print()

    print("\nBatch Actions:")
    print("  all-intended: Accept all intended labels")
    print("  all-model: Accept all model labels")
    print("  interactive: Switch to interactive mode")
    print("  q: Quit")

    cmd = input("\nBatch decision: ").strip().lower()

    if cmd == 'all-intended':
        return [{**row, 'human_label': row['intended_class'], 'review_decision': 'batch_kept_intended'} for row in records]
    elif cmd == 'all-model':
        return [{**row, 'human_label': row['labeled_class'], 'review_decision': 'batch_kept_model'} for row in records]
    elif cmd == 'interactive':
        return review_interactive(records, category_name)
    else:
        return []

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Review disagreements interactively')
    parser.add_argument('--category', help='Category to review', default=None)
    parser.add_argument('--batch', action='store_true', help='Batch review mode')
    parser.add_argument('--list-categories', action='store_true', help='List available categories')
    args = parser.parse_args()

    # Load categories
    with open('data/review_categories.json', 'r') as f:
        categories = json.load(f)

    if args.list_categories:
        print("\nAvailable categories:")
        for cat, items in categories.items():
            print(f"  {cat}: {len(items)} items")
        return

    # Select category
    if args.category:
        if args.category not in categories:
            print(f"Unknown category: {args.category}")
            print("Use --list-categories to see available categories")
            return
        records = categories[args.category]
        cat_name = args.category
    else:
        # Default to high confidence model likely right
        cat_name = 'high_confidence_model_likely_right'
        records = categories[cat_name]

    # Review
    if args.batch:
        results = batch_review(records, cat_name)
    else:
        results = review_interactive(records, cat_name)

    # Save results
    if results:
        output_file = f'data/reviewed_{cat_name}.csv'
        fieldnames = list(results[0].keys())
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✓ Saved {len(results)} reviewed records to {output_file}")

        # Show summary
        decisions = {}
        for r in results:
            decisions[r['review_decision']] = decisions.get(r['review_decision'], 0) + 1

        print("\nReview Summary:")
        for decision, count in decisions.items():
            print(f"  {decision}: {count}")

if __name__ == '__main__':
    main()
