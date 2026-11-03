#!/usr/bin/env python3
"""
Automated disagreement review based on taxonomy rules and patterns.
Auto-resolves high-confidence cases, flags uncertain ones for human review.
"""

import csv
import json
import re
import sys
import os
from pathlib import Path
from collections import defaultdict

# Navigate up to project root
project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)

# Add src to path
sys.path.insert(0, str(project_root / 'src'))
from taxonomy import TAXONOMY, LABELS

def is_birth_date(column_name):
    """Check if column name clearly indicates date of birth."""
    birth_indicators = ['birth', 'dob', 'bday', 'birthday', 'born']
    col_lower = column_name.lower()
    return any(indicator in col_lower for indicator in birth_indicators)

def is_session_id(column_name, data_type):
    """Check if column is a session identifier."""
    session_indicators = ['session', 'sess']
    col_lower = column_name.lower()
    is_session = any(indicator in col_lower for indicator in session_indicators)
    is_id_type = 'uuid' in data_type.lower() or 'key' in col_lower or 'id' in col_lower
    return is_session and is_id_type

def is_account_number(column_name):
    """Check if column is an account number."""
    return 'acct' in column_name.lower() and ('nbr' in column_name.lower() or
                                                'num' in column_name.lower() or
                                                'key' in column_name.lower() or
                                                'id' in column_name.lower())

def is_segment_metadata(table_name, column_name):
    """Check if this is segment/audience metadata, not individual demographics."""
    table_lower = table_name.lower()
    col_lower = column_name.lower()

    # Table indicators
    segment_tables = ['seg', 'audience', 'aud', 'ad_seg', 'target', 'cohort']
    is_segment_table = any(seg in table_lower for seg in segment_tables)

    # Column indicators for aggregate/target demographics
    aggregate_indicators = ['_bkt', '_grp', '_band', '_range', '_catg', '_cat_cd']
    is_aggregate = any(ind in col_lower for ind in aggregate_indicators)

    return is_segment_table and is_aggregate

def is_business_location(table_name, column_name):
    """Check if address is for a business, not a person."""
    table_lower = table_name.lower()
    col_lower = column_name.lower()

    business_indicators = ['operator', 'vendor', 'org', 'branch', 'office',
                          'warehouse', 'store', 'facility', 'venue']
    return any(ind in table_lower for ind in business_indicators)

def auto_review_record(row):
    """
    Auto-review a disagreement record and return (decision, confidence, reason).

    Returns:
        (label, confidence, reason) where confidence is 'high', 'medium', 'low', or 'uncertain'
    """
    table = row['table_name']
    column = row['column_name']
    col_lower = column.lower()
    dtype = row['data_type']
    intended = row['intended_class']
    labeled = row['labeled_class']
    mode = row['mode']

    # Pattern 1: Birth dates mislabeled as demographic_attribute
    if intended == 'demographic_attribute' and labeled == 'date_of_birth' and is_birth_date(column):
        return ('date_of_birth', 'high', 'Column name clearly indicates date of birth')

    # Pattern 2: Session IDs marked as non_pii but should be online_identifier
    if intended == 'non_pii' and labeled == 'online_identifier' and is_session_id(column, dtype):
        return ('online_identifier', 'high', 'Session IDs are online identifiers per taxonomy')

    # Pattern 3: Account numbers
    if is_account_number(column):
        if intended == 'non_pii' and labeled == 'financial_account':
            return ('financial_account', 'high', 'Account numbers are financial_account per taxonomy')
        if intended == 'financial_account' and labeled == 'non_pii':
            return ('financial_account', 'medium', 'acct pattern suggests financial_account')

    # Pattern 4: Segment/audience metadata (not individual demographics)
    if is_segment_metadata(table, column):
        if labeled == 'demographic_attribute' and intended == 'non_pii':
            return ('non_pii', 'high', 'Segment metadata, not individual demographic')

    # Pattern 5: Business locations (not personal addresses)
    if intended == 'physical_address' and labeled == 'non_pii' and is_business_location(table, column):
        return ('non_pii', 'medium', 'Business location, not personal address')

    # Pattern 6: Hard negatives where model correctly identified PII
    if mode == 'hard_negative' and intended == 'non_pii' and labeled != 'non_pii':
        # Check if the labeled class makes sense
        source = row.get('source_class', '')
        if labeled == source:
            return (labeled, 'high', f'Hard negative correctly identified as {labeled}')

    # NEW Pattern 7: Online identifiers being labeled as non_pii
    if intended == 'online_identifier' and labeled == 'non_pii':
        # Check for clear online identifier patterns
        online_patterns = ['mac_addr', 'mac', 'device_id', 'device_serial', 'uuid', 'guid',
                          'instance_id', 'subscriber_key', 'conn_key', 'token', 'cookie',
                          'tracking', 'session', 'ws_conn', 'app_install']

        if any(pattern in col_lower for pattern in online_patterns):
            return ('online_identifier', 'high', f'Clear online identifier pattern in column name')

    # NEW Pattern 8: Phone numbers being labeled as non_pii
    if intended == 'phone' and labeled == 'non_pii':
        phone_patterns = ['phone', 'tel', 'fax', 'mobile', 'ph_', '_ph']
        if any(pattern in col_lower for pattern in phone_patterns):
            return ('phone', 'high', 'Clear phone number pattern in column name')

    # NEW Pattern 9: Financial accounts being labeled as non_pii
    if intended == 'financial_account' and labeled == 'non_pii':
        fin_patterns = ['card', 'payment', 'pmt_', 'fund_src', 'expir', 'cvv', 'iban']
        if any(pattern in col_lower for pattern in fin_patterns):
            return ('financial_account', 'medium', 'Financial account pattern in column name')

    # NEW Pattern 10: Physical addresses being labeled as non_pii
    if intended == 'physical_address' and labeled == 'non_pii':
        if not is_business_location(table, column):
            addr_patterns = ['addr', 'address', 'street', 'postal', 'zip', 'city', 'province', 'state']
            if any(pattern in col_lower for pattern in addr_patterns):
                return ('physical_address', 'medium', 'Address pattern in column name')

    # Pattern 11: Generic table + vague column = trust model with caution
    if mode == 'ambiguous_table':
        # If model has high confidence, lean towards it
        if row['confidence'] == 'high':
            return (labeled, 'medium', 'Ambiguous table, trusting high-confidence model')
        else:
            return (None, 'uncertain', 'Ambiguous table and low model confidence')

    # Pattern 12: Technical suffixes - if model identified PII despite suffix, likely correct
    if mode == 'abbreviated_legacy':
        # _key, _id, _nbr, _cd patterns
        if labeled != 'non_pii' and intended != 'non_pii':
            # Both think it's PII, just different classes
            return (None, 'uncertain', 'Both identified as PII but different classes')
        if labeled != 'non_pii' and intended == 'non_pii':
            # Model found PII through the technical suffix
            return (labeled, 'medium', 'Model identified PII despite technical suffix')

    # Pattern 13: High confidence disagreements
    if row['confidence'] == 'high':
        # Model is very confident - if it disagrees with intended, model might be right
        if labeled == 'non_pii' and intended != 'non_pii':
            # Model confidently says non_pii
            return (None, 'uncertain', 'High confidence non_pii - needs human judgment')
        else:
            return (labeled, 'medium', 'Model has high confidence')

    # Default: uncertain
    return (None, 'uncertain', 'No clear pattern match')

def main():
    data_dir = Path('data')

    # Load resolved disagreements if they exist
    resolved_file = data_dir / 'synthetic_pii_columns_disagreements_resolved.csv'
    already_resolved = set()

    if resolved_file.exists():
        with open(resolved_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get('human_label'):
                    already_resolved.add((row['table_name'], row['column_name']))

    print(f"Already resolved: {len(already_resolved)} records\n")

    # Load all disagreements
    with open(data_dir / 'synthetic_pii_columns_disagreements.csv', 'r') as f:
        reader = csv.DictReader(f)
        disagreements = list(reader)

    # Filter to unresolved
    unresolved = [row for row in disagreements
                  if (row['table_name'], row['column_name']) not in already_resolved]

    print(f"Unresolved disagreements: {len(unresolved)}\n")
    print("=" * 80)
    print("AUTO-REVIEWING UNRESOLVED DISAGREEMENTS")
    print("=" * 80)

    # Auto-review
    results = {
        'high_confidence': [],
        'medium_confidence': [],
        'uncertain': []
    }

    for row in unresolved:
        label, confidence, reason = auto_review_record(row)

        review_result = {
            **row,
            'auto_label': label if label else 'NEEDS_REVIEW',
            'auto_confidence': confidence,
            'auto_reason': reason
        }

        if confidence == 'high':
            results['high_confidence'].append(review_result)
        elif confidence == 'medium':
            results['medium_confidence'].append(review_result)
        else:
            results['uncertain'].append(review_result)

    # Print summary
    print(f"\nAuto-review results:")
    print(f"  High confidence: {len(results['high_confidence'])} (will auto-resolve)")
    print(f"  Medium confidence: {len(results['medium_confidence'])} (will auto-resolve)")
    print(f"  Uncertain: {len(results['uncertain'])} (need human review)")

    # Save auto-resolved to reviewed file
    auto_resolved = results['high_confidence'] + results['medium_confidence']

    if auto_resolved:
        # Convert to format expected by merge script
        for row in auto_resolved:
            row['human_label'] = row['auto_label']
            row['review_decision'] = f"auto_{row['auto_confidence']}_confidence"

        output_file = data_dir / 'reviewed_auto_resolved.csv'
        fieldnames = list(auto_resolved[0].keys())  # Get fieldnames after adding new columns

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(auto_resolved)

        print(f"\n✓ Saved {len(auto_resolved)} auto-resolved to {output_file}")

    # Save uncertain cases for human review
    if results['uncertain']:
        output_file = data_dir / 'uncertain_needs_human_review.csv'
        fieldnames = list(results['uncertain'][0].keys())

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results['uncertain'])

        print(f"✓ Saved {len(results['uncertain'])} uncertain cases to {output_file}")

        # Show some examples
        print(f"\n{'=' * 80}")
        print("UNCERTAIN CASES NEEDING YOUR REVIEW (showing first 10)")
        print("=" * 80)

        for i, row in enumerate(results['uncertain'][:10], 1):
            print(f"\n{i}. {row['table_name']}.{row['column_name']} ({row['data_type']})")
            print(f"   Intended: {row['intended_class']} → Model: {row['labeled_class']} [{row['confidence']}]")
            print(f"   Reason: {row['auto_reason']}")

    print(f"\n{'=' * 80}")
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Run merge to apply auto-resolved labels:")
    print("   python merge_reviews.py")
    print(f"\n2. Review the {len(results['uncertain'])} uncertain cases manually")
    print("   (saved in uncertain_needs_human_review.csv)")

if __name__ == '__main__':
    main()
