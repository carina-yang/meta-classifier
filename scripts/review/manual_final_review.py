#!/usr/bin/env python3
"""
Final manual review decisions for the remaining 65 uncertain cases.
Based on taxonomy rules and column name semantics.
"""

import csv
import os
from pathlib import Path

# Manual decisions for each uncertain case
# Format: (table_name, column_name): (decision, confidence, reason)
MANUAL_DECISIONS = {
    # 1. Language code - demographic attribute
    ('ms_contact_extra', 'native_language_cd'): ('demographic_attribute', 'high', 'Language/nationality is demographic'),

    # 2. Guarantor expiry date - financial_account
    ('dep_guar', 'guar_exp_dt'): ('financial_account', 'medium', 'Guarantor expiry relates to financial accounts'),

    # 3. DOB flag - metadata ABOUT DOB, not the DOB itself
    ('CustomerPrefs', 'DobFlg'): ('non_pii', 'high', 'Flag/indicator about DOB, not the DOB value itself'),

    # 4. Primary address name - physical_address (address component)
    ('cust_profile', 'col_prim_addr_nm'): ('physical_address', 'medium', 'Address name field'),

    # 5. Screen name - online_identifier
    ('don_form_audit', 'col_scrn_nm'): ('online_identifier', 'high', 'Screen name is an online identifier'),

    # 6. Owner middle name - person_name
    ('stg_acct_owner_clean', 'owner_mdl_nm'): ('person_name', 'high', 'Middle name (mdl) is part of person name'),

    # 7. Reset token key - online_identifier
    ('legacy_pwd_reset', 'reset_tok_key'): ('online_identifier', 'high', 'Password reset tokens are identifiers'),

    # 8. Install ID - online_identifier
    ('src_app_installs', 'install_id'): ('online_identifier', 'high', 'Installation IDs track devices'),

    # 9. Authorized user name - online_identifier (username context)
    ('cust_ref_tbl', 'auth_usr_nm_2'): ('online_identifier', 'medium', 'Authorized username for system access'),

    # 10. Subscriber identifier name - online_identifier
    ('abonne_ref', 'ident_abonne_nm'): ('online_identifier', 'medium', 'Subscriber identifier (French: abonne)'),

    # 11. Expiry date in license app - non_pii (license expiry, not financial)
    ('legacy_licence_app', 'expiry_dt'): ('non_pii', 'high', 'License expiry date, not financial card expiry'),

    # 12. Religion code - demographic_attribute
    ('empMstr_new', 'relgnCd'): ('demographic_attribute', 'high', 'Religion code is demographic per taxonomy'),

    # 13. License name - government_id
    ('usr_ticket_holder_bak', 'lic_nm'): ('government_id', 'medium', 'License name field (drivers license)'),

    # 14. External mail address - email (MailAddr usually means email in modern systems)
    ('src_customer_ext', 'extMailAddr'): ('email', 'high', 'MailAddr typically means email address'),

    # 15. Secondary address name - physical_address
    ('cust_profile', 'col_sec_addr_nm'): ('physical_address', 'medium', 'Secondary address name field'),

    # 16. Refresh token old - online_identifier
    ('api_client_cred', 'refresh_tok_old'): ('online_identifier', 'high', 'Refresh tokens are online identifiers'),

    # 17. Member gender flag - demographic_attribute
    ('loyalty_member', 'member_gender_flg'): ('demographic_attribute', 'high', 'Gender is demographic per taxonomy'),

    # 18. Building state code - physical_address
    ('bldg_mstr', 'fld_bldg_st_cd'): ('physical_address', 'medium', 'State code (st_cd) is address component'),

    # 19. Smartcard number - financial_account
    ('cas_card_map', 'smartcard_nbr'): ('financial_account', 'high', 'Smart cards used for payment'),

    # 20. Service unit ID - physical_address (service location unit)
    ('service_location', 'svc_unit_id'): ('physical_address', 'medium', 'Service location unit identifier'),

    # 21. Display name clean - online_identifier
    ('src_leaderboard_snap', 'display_nm_clean'): ('online_identifier', 'high', 'Display names are online identifiers'),

    # 22. Gateway session key - online_identifier
    ('TblPayrollGateway', 'GwSessKey'): ('online_identifier', 'high', 'Gateway session key is identifier'),

    # 23. Unit number in housing - physical_address
    ('hsgUnitInv', 'unitNbr'): ('physical_address', 'medium', 'Housing unit number is address component'),

    # 24. Assigned user name - online_identifier
    ('Seats', 'assigned_user_name'): ('online_identifier', 'high', 'Username assignment'),

    # 25. Email flag source - non_pii
    ('pto_accrual', 'accr_email_flg_src'): ('non_pii', 'high', 'Flag about email, not email itself'),

    # 26. Driver license flag ID - non_pii
    ('POL_UNDERWRITE_APP', 'uw_applcnt_drvr_lic_flg_id'): ('non_pii', 'high', 'Flag/indicator about license, not license itself'),

    # 27. Reset token raw - online_identifier
    ('TblPwResetReq', 'ResetTokn_raw'): ('online_identifier', 'high', 'Password reset token'),

    # 28. Beneficiary preferred name - person_name
    ('benefit_enroll_x', 'bnf_pref_nm'): ('person_name', 'high', 'Preferred name is person name'),

    # 29. Driver license series number - government_id
    ('drv_lic_series_ref', 'drv_lic_ser_nbr'): ('government_id', 'medium', 'Driver license series/reference'),

    # 30. Device name - online_identifier
    ('usr_device_bind', 'dvc_nm'): ('online_identifier', 'high', 'Device name identifies device'),

    # 31. Web visitor ID - online_identifier
    ('stg_web_visitor', 'stg_web_visitor_id'): ('online_identifier', 'high', 'Visitor ID is online identifier'),

    # 32. DOB flag source - date_of_birth
    ('src_retail_cif', 'c_dob_flg_src'): ('date_of_birth', 'medium', 'DOB flag that contains birth date'),

    # 33. Room number - physical_address
    ('legacy_dorm_rm', 'rm_nbr'): ('physical_address', 'medium', 'Room number in dorm is address component'),

    # 34. API access token key - online_identifier
    ('api_client_cred', 'api_access_tok_key'): ('online_identifier', 'high', 'API access token is identifier'),

    # 35. Requester phone name flag - non_pii
    ('tmp_maint_req_raw', 'reqstr_ph_nm_flg'): ('non_pii', 'medium', 'Flag about phone/name, unclear'),

    # 36. DOB indicator - date_of_birth
    ('CRD_CARDHLDR', 'fld_dob_ind'): ('date_of_birth', 'high', 'DOB indicator field with date type'),

    # 37. Mail address - email
    ('usr_profile_clean', 'mail_addr'): ('email', 'high', 'mail_addr typically means email'),

    # 38. Primary government ID name - government_id
    ('primary_id_doc', 'primaryGovtIdNm'): ('government_id', 'high', 'Government ID name field'),

    # 39. Adjuster government ID flag - non_pii
    ('adjstr_assgn', 'adj_govt_id_flg_v2'): ('non_pii', 'high', 'Flag about govt ID, not the ID itself'),

    # 40. Guest DOB indicator text - date_of_birth
    ('CartSession', 'GuestDobIndTxt'): ('date_of_birth', 'medium', 'DOB indicator text'),

    # 41. SSN indicator - non_pii
    ('ADJ_NOTE_HIST', 'adj_note_ssn_ind'): ('non_pii', 'high', 'Indicator about SSN, not SSN itself'),

    # 42. Recovery agent national ID name - government_id
    ('RCVRY_AGENT_MAP', 'rcvry_agt_nat_id_nm'): ('government_id', 'medium', 'National ID name'),

    # 43. Preference birthday text - date_of_birth
    ('user_consent_preferences', 'pref_birthday_txt'): ('date_of_birth', 'high', 'Birthday preference field'),

    # 44. Attorney national ID flag - non_pii
    ('src_legal_rep', 'src_atty_natl_id_flg'): ('non_pii', 'high', 'Flag about national ID, not ID itself'),

    # 45. Preference birth year - date_of_birth
    ('tenant_pref', 'pref_birth_yr'): ('date_of_birth', 'high', 'Birth year preference'),

    # 46. Screen name - online_identifier
    ('adjuster_session_screen', 'screen_nm'): ('online_identifier', 'high', 'Screen name identifier'),

    # 47. API client ID key - online_identifier
    ('payroll_api_client', 'api_client_id_key'): ('online_identifier', 'high', 'API client identifier'),

    # 48. ID card SSN flag code - non_pii
    ('src_refund_request_raw', 'idcard_ssn_flg_cd'): ('non_pii', 'high', 'Flag about SSN, not SSN itself'),

    # 49. Text body - non_pii (generic text field)
    ('jobctr_seeker', 'txt_bdy'): ('non_pii', 'high', 'Generic text body field'),

    # 50. Session key - online_identifier
    ('stg_session_log_clean', 'sess_key'): ('online_identifier', 'high', 'Session key is identifier'),

    # 51. DOB indicator - date_of_birth
    ('referral_lead', 'ref_dob_ind'): ('date_of_birth', 'medium', 'DOB indicator field'),

    # 52. Set-top box serial number - online_identifier
    ('stb_dvc_map', 'stb_serial_nbr'): ('online_identifier', 'high', 'Device serial number'),

    # 53. Preference name - person_name
    ('CUST_PREF', 'pref_nm'): ('person_name', 'medium', 'Preference name (likely person name)'),

    # 54. Province name text - physical_address
    ('player_kyc_doc', 'txt_prov_nm'): ('physical_address', 'medium', 'Province name is address component'),

    # 55. Scanned passport name - government_id
    ('src_gate_scan_log', 'scanned_passport_nm'): ('government_id', 'high', 'Passport name from scan'),

    # 56. DOB flag - non_pii
    ('tmp_viewer_pref', 'dobFlg'): ('non_pii', 'high', 'Flag about DOB, not DOB itself'),

    # 57. Thermostat serial number - online_identifier
    ('thermostat_enrol', 'thermostat_serial_nbr'): ('online_identifier', 'high', 'Device serial number'),

    # 58. API client identifier key - online_identifier
    ('partner_api_clients', 'api_client_identifier_key'): ('online_identifier', 'high', 'API client identifier'),

    # 59. Portal identifier key - online_identifier
    ('emp_tax_forms_w4', 'w4_portal_ident_key'): ('online_identifier', 'high', 'Portal identifier'),

    # 60. Portal user key - online_identifier
    ('tour_op_portal', 'portal_usr_key'): ('online_identifier', 'high', 'Portal user identifier'),

    # 61. Name ID text - online_identifier
    ('saml_assertions', 'name_id_txt'): ('online_identifier', 'high', 'SAML name ID (federated identity)'),

    # 62. Quote origin state code - physical_address
    ('freight_quote_raw', 'quoteOrigStCd'): ('physical_address', 'medium', 'Origin state code is address'),

    # 63. DOB stamp - date_of_birth
    ('tenant_usr', 'dob_stamp'): ('date_of_birth', 'high', 'DOB timestamp'),

    # 64. Temporary session key - online_identifier
    ('tmp_sess_cache', 'str_tmp_sess_key'): ('online_identifier', 'high', 'Session key identifier'),

    # 65. Player name text - online_identifier
    ('legacy_player_xref', 'plyr_nm_txt'): ('online_identifier', 'medium', 'Player name in gaming context (handle)'),
}

def main():
    # Navigate to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    os.chdir(project_root)

    data_dir = Path('data')

    # Load uncertain cases
    with open(data_dir / 'uncertain_needs_human_review.csv', 'r') as f:
        reader = csv.DictReader(f)
        uncertain = list(reader)

    print(f"Total uncertain cases: {len(uncertain)}")
    print(f"Manual decisions prepared: {len(MANUAL_DECISIONS)}\n")

    # Apply manual decisions
    resolved = []
    for row in uncertain:
        key = (row['table_name'], row['column_name'])
        if key in MANUAL_DECISIONS:
            decision, confidence, reason = MANUAL_DECISIONS[key]
            row['human_label'] = decision
            row['review_decision'] = f'manual_{confidence}_confidence'
            row['manual_reason'] = reason
            resolved.append(row)

    print(f"Resolved: {len(resolved)}")

    # Save
    if resolved:
        output_file = data_dir / 'reviewed_manual_final.csv'
        fieldnames = list(resolved[0].keys())

        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(resolved)

        print(f"✓ Saved to {output_file}")

        # Summary
        label_counts = {}
        for row in resolved:
            label = row['human_label']
            label_counts[label] = label_counts.get(label, 0) + 1

        print("\nLabel distribution:")
        for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
            print(f"  {label:30s}: {count:3d}")

if __name__ == '__main__':
    main()
