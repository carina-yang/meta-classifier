#!/usr/bin/env python3
"""
Merge all reviewed disagreements into a final human-labeled dataset.

This script:
1. Loads all reviewed_*.csv files
2. Combines them with the original synthetic data
3. Creates a final training dataset with human-verified labels
"""

import csv
import os
from pathlib import Path
from collections import defaultdict

def main():
    # Navigate to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    data_dir = Path('data')

    # Find all reviewed files
    reviewed_files = list(data_dir.glob('reviewed_*.csv'))

    if not reviewed_files:
        print("No reviewed files found!")
        print("Review some disagreements first using review_disagreements.py")
        return

    print(f"Found {len(reviewed_files)} review file(s):")
    for f in reviewed_files:
        print(f"  - {f.name}")

    # Load all reviewed records
    reviewed = {}  # key: (table_name, column_name) -> human_label
    review_stats = defaultdict(int)

    for file in reviewed_files:
        with open(file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['table_name'], row['column_name'])
                reviewed[key] = row['human_label']
                review_stats[row['review_decision']] += 1

    print(f"\nTotal reviewed records: {len(reviewed)}")
    print("\nReview decisions:")
    for decision, count in sorted(review_stats.items()):
        print(f"  {decision}: {count}")

    # Load disagreements file (the records we reviewed)
    with open(data_dir / 'synthetic_pii_columns_disagreements.csv', 'r') as f:
        reader = csv.DictReader(f)
        disagreements = list(reader)

    # Update with human labels
    updated_count = 0
    for row in disagreements:
        key = (row['table_name'], row['column_name'])
        if key in reviewed:
            # Add human_label and review_decision columns
            row['human_label'] = reviewed[key]
            # Check if human agrees with model or intended
            if row['human_label'] == row['labeled_class']:
                row['resolution'] = 'agreed_with_model'
            elif row['human_label'] == row['intended_class']:
                row['resolution'] = 'agreed_with_intended'
            else:
                row['resolution'] = 'manual_override'
            updated_count += 1

    # Write corrected disagreements file
    output_file = data_dir / 'synthetic_pii_columns_disagreements_resolved.csv'

    # Add new columns if they don't exist
    fieldnames = list(disagreements[0].keys())
    if 'human_label' not in fieldnames:
        fieldnames.extend(['human_label', 'resolution'])

    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(disagreements)

    print(f"\n✓ Updated {updated_count} records with human labels")
    print(f"✓ Saved to {output_file}")

    # Show resolution statistics
    resolution_dist = defaultdict(int)
    label_dist = defaultdict(int)

    for row in disagreements:
        if 'human_label' in row:
            label_dist[row['human_label']] += 1
            if 'resolution' in row:
                resolution_dist[row['resolution']] += 1

    if resolution_dist:
        print(f"\nResolution breakdown:")
        for resolution, count in sorted(resolution_dist.items(), key=lambda x: -x[1]):
            pct = count / updated_count * 100 if updated_count > 0 else 0
            print(f"  {resolution:30s}: {count:5d} ({pct:5.1f}%)")

    if label_dist:
        print(f"\nHuman-labeled class distribution (resolved only):")
        for label, count in sorted(label_dist.items(), key=lambda x: -x[1]):
            pct = count / updated_count * 100 if updated_count > 0 else 0
            print(f"  {label:30s}: {count:5d} ({pct:5.1f}%)")

    print(f"\nTotal disagreements: {len(disagreements)}")
    print(f"Resolved by human review: {updated_count} ({updated_count/len(disagreements)*100:.1f}%)")
    print(f"Still unresolved: {len(disagreements) - updated_count} ({(len(disagreements) - updated_count)/len(disagreements)*100:.1f}%)")

if __name__ == '__main__':
    main()
