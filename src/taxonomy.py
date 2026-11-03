"""
taxonomy.py
-----------
Defines the 10-class PII taxonomy used throughout the pipeline.

This module serves as the single source of truth for:
- Class definitions and examples (TAXONOMY dict)
- Fixed label order (LABELS list) - determines stable integer encoding
- Include/exclude patterns for each class
- Labeling guidance for LLM-based data generation

Classes:
  person_name, email, phone, physical_address, government_id, date_of_birth,
  financial_account, online_identifier, demographic_attribute, non_pii

Each class specifies:
- definition: What the class represents
- include: Example column names that belong to this class
- exclude: Counter-examples with their correct class
- fallback: Default rules for ambiguous cases
- notes: Additional clarifications

Used by synthetic_data.py (prompts), label_encoding.py (integer IDs), and
data_cleaning.py (validation).
"""
from typing import Dict, List

NON_PII = "non_pii"

TAXONOMY: Dict[str, dict] = {
    "person_name": {
        "definition": "a column holding all or part of an individual's name.",
        "include": ["first_name", "last_name", "full_name", "middle_name", "surname",
                    "maiden_name", "cust_nm", "display_name", "nickname",
                    "first_nm", "last_nm", "CUST_FIRST_NAME", "customer_last_name",
                    "name_first", "name_last", "primary_contact_name", "billing_name",
                    "frst_nm", "lst_nm", "full_nm", "legal_name", "preferred_name"],
        "exclude": [("company_name", "non_pii"), ("product_name", "non_pii"),
                    ("file_name", "non_pii"), ("table_name", "non_pii"),
                    ("username", "online_identifier")],
        "fallback": "a bare `name` with an uninformative table -> person_name",
        "notes": [],
    },
    "email": {
        "definition": "an email address.",
        "include": ["email", "contact_email", "EmailAddr", "eml",
                    "email_1", "email_2", "email_primary", "email_secondary",
                    "email_addr", "email_address", "cust_email", "user_eml",
                    "primary_email_address", "legacy_email", "email_txt",
                    "billing_email", "shipping_email", "CUST_EML", "usr_email",
                    "email_addr_1", "email_addr_2", "contact_eml", "old_email"],
        "exclude": [("email_template_id", "non_pii"), ("email_sent_count", "non_pii"),
                    ("email_verified_flag", "non_pii")],
        "fallback": None,
        "notes": [],
    },
    "phone": {
        "definition": "a telephone number.",
        "include": ["phone", "mobile_no", "tel_1", "fax",
                    "phone_number", "phone_1", "phone_2", "phone_primary",
                    "ph_nbr", "phone_nbr", "telephone", "tel_no", "tel_nbr",
                    "contact_phone", "mobile_phone", "home_phone", "work_phone",
                    "CNTCT_TEL", "phone_number_1", "phone_number_2", "legacy_phone",
                    "cust_phone", "primary_phone_nbr", "tel", "fax_number"],
        "exclude": [("phone_model", "non_pii"), ("phone_type", "non_pii"),
                    ("phone_carrier", "non_pii"), ("extension_count", "non_pii")],
        "fallback": None,
        "notes": [],
    },
    "physical_address": {
        "definition": "a street address or a component that locates a person.",
        "include": ["street_addr", "unit_number", "postal_code", "zip", "city",
                    "province", "state",
                    "address", "addr", "address_line1", "address_line2", "addr_line_1",
                    "street_address", "mailing_address", "billing_address", "shipping_addr",
                    "ADDR_LN1", "ADDR_LN2", "street_addr_1", "zip_code", "postal_cd",
                    "city_name", "state_cd", "province_cd", "address_txt", "legacy_address",
                    "cust_addr", "primary_address", "addr_1", "addr_2"],
        "exclude": [("warehouse_location", "non_pii"), ("server_region", "non_pii"),
                    ("country", "non_pii")],
        "fallback": "a bare `address` or `addr` with an uninformative table -> physical_address",
        "notes": [],
    },
    "government_id": {
        "definition": "an identifier issued by a government to an individual.",
        "include": ["ssn", "sin", "passport_no", "drivers_lic", "tax_id", "national_id",
                    "social_security_number", "passport_number", "drivers_license",
                    "driver_license_no", "SSN", "tax_identification_number", "govt_id",
                    "national_id_number", "passport_nbr", "license_number", "dl_number",
                    "taxpayer_id", "social_sec_no", "passport_id"],
        "exclude": [("order_id", "non_pii"), ("ticket_number", "non_pii")],
        "fallback": None,
        "notes": ["internal identifiers such as order_id and ticket_number are not government ids"],
    },
    "date_of_birth": {
        "definition": "a person's birth date or birth year.",
        "include": ["dob", "birth_date", "date_of_birth", "birthday", "birth_year",
                    "DOB", "birthdate", "birth_dt", "dob_dt", "date_birth",
                    "birth_day", "birth_yr", "yr_of_birth", "bday", "cust_dob",
                    "user_birth_date", "person_dob"],
        "exclude": [("created_date", "non_pii"), ("expiry_date", "non_pii"),
                    ("hire_date", "non_pii"), ("age", "demographic_attribute")],
        "fallback": "a bare `date` -> non_pii",
        "notes": [],
    },
    "financial_account": {
        "definition": "an identifier or credential for an individual's financial account or payment card.",
        "include": ["card_number", "iban", "acct_no", "cvv", "card_expiry",
                    "account_number", "credit_card_number", "cc_number", "card_nbr",
                    "bank_account_number", "bank_acct_no", "acct_nbr", "IBAN",
                    "account_no", "credit_card_no", "debit_card_number", "cc_num",
                    "cvv_code", "card_expiry_date", "expiry_dt", "payment_card_number"],
        "exclude": [("account_type", "non_pii"), ("branch_code", "non_pii"),
                    ("currency", "non_pii"), ("balance", "non_pii"),
                    ("routing_number", "non_pii"), ("card_brand", "non_pii"),
                    ("card_network", "non_pii")],
        "fallback": None,
        "notes": ["routing_number is institution-level -> non_pii",
                  "card_brand / card_network (Visa/Mastercard) are network identifiers, not card numbers -> non_pii"],
    },
    "online_identifier": {
        "definition": "a handle or technical identifier tied to a person or device.",
        "include": ["ip_address", "device_id", "username", "user_handle", "mac_addr",
                    "IP_address", "ip_addr", "device_identifier", "mac_address",
                    "user_name", "login_name", "screen_name", "handle", "device_uuid",
                    "ip", "MAC_addr", "device_token", "browser_id", "session_id",
                    "cookie_id", "client_ip", "usr_name"],
        "exclude": [("user_agent", "non_pii"), ("session_count", "non_pii"),
                    ("ip_version", "non_pii"), ("user_id", "non_pii"),
                    ("customer_id", "non_pii")],
        "fallback": None,
        "notes": ["internal surrogate keys such as user_id and customer_id -> non_pii"],
    },
    "demographic_attribute": {
        "definition": "a column holding a demographic characteristic of a person.",
        "include": ["gender", "sex", "ethnicity", "race", "marital_status", "nationality",
                    "citizenship", "religion", "age", "age_group", "age_band",
                    "gender_txt", "sex_cd", "ethnic_group", "race_cd", "marital_stat",
                    "country_of_birth", "citizenship_status", "age_range", "age_bracket",
                    "customer_age", "user_gender", "person_sex", "cust_age"],
        "exclude": [("gender_id", "non_pii"), ("religion_code", "non_pii"),
                    ("gender_neutral_flag", "non_pii"), ("ethnicity_field_enabled", "non_pii"),
                    ("marital_status_lookup_id", "non_pii"), ("account_age_days", "non_pii"),
                    ("device_age", "non_pii"), ("vehicle_age", "non_pii"),
                    ("age_of_record", "non_pii"), ("min_age_flag", "non_pii"),
                    ("age_limit", "non_pii")],
        "fallback": "a bare `race` or `religion` -> demographic_attribute",
        "notes": ["age of a non-person thing (account_age_days, device_age, vehicle_age, ...) -> non_pii",
                  "OUT OF SCOPE: health information (diagnosis, medication) is not covered by this dataset; never use such columns"],
    },
    "non_pii": {
        "definition": "everything else, including all hard negatives above.",
        "include": ["price", "status", "quantity", "created_at", "country", "id",
                    "business_license_id", "software_license_key",
                    "order_id", "product_id", "sku", "amount", "total_amt", "qty",
                    "created_date", "updated_at", "modified_dt", "status_cd", "status_flag",
                    "is_active", "is_deleted", "country_cd", "region", "category",
                    "product_name", "description", "notes", "comment_txt"],
        "exclude": [],
        "fallback": None,
        "notes": [],
    },
}

LABELS: List[str] = list(TAXONOMY.keys())  # fixed order -> fixed integer ids
PII_LABELS: List[str] = [l for l in LABELS if l != NON_PII]

LABELING_GUIDANCE = """\
Labeling rules:
- Classify the COLUMN using column_name as the main signal; table_name and data_type are context.
- Distinguish PII values from metadata ABOUT PII:
  • METADATA words indicate properties/types/counts (→ non_pii): template, type, kind, model, brand,
    carrier, count, status, category, format, style, preference, enabled, verified
    Examples: email_template_id, phone_type_cd, gender_id, card_brand, email_sent_count, phone_carrier
  • CONTEXT words describe which/whose PII (→ still PII): billing, shipping, home, work, primary,
    secondary, old, new, legacy, contact, customer, user, personal
    Examples: billing_email → email, home_phone → phone, primary_ssn_cd → government_id
  • Technical suffixes are just naming conventions (ignore them): _key, _cd, _id, _nbr, _txt, _flg, _dt, _nm
    Examples: email_key → email, ssn_cd → government_id, phone_nbr → phone
- IMPORTANT: Storage format words do NOT change classification (→ still PII): token, tok, tkn, hash,
  encrypted, enc, mask, vault, obfuscated refer to how PII is stored, not what it is.
  Examples: cc_tok_id → financial_account, ssn_mask → government_id, email_hash → email,
  vault_card_token_nbr → financial_account, encrypted_ssn → government_id
- Pattern: If column contains a metadata word, it's usually non_pii. If it's just PII word + context/suffix,
  classify as the PII class. The Include/Exclude lists show examples of both patterns.
- Apply the Fallback default rules when the table name is uninformative (e.g. tbl_01, data, tmp_load).
- Exactly one label per column. If unsure between a PII class and non_pii, prefer the taxonomy's Exclude lists."""


def render_taxonomy() -> str:
    """Render the taxonomy as markdown for use inside prompts."""
    out = []
    for label, spec in TAXONOMY.items():
        out.append(f"### {label}")
        out.append(f"- Definition: {spec['definition']}")
        out.append("- Include: " + ", ".join(f"`{x}`" for x in spec["include"]))
        if spec["exclude"]:
            ex = []
            for name, route in spec["exclude"]:
                ex.append(f"`{name}`" + ("" if route == NON_PII else f" (-> `{route}`)"))
            out.append("- Exclude: " + ", ".join(ex))
        if spec["fallback"]:
            out.append(f"- Fallback default: {spec['fallback']}")
        for note in spec["notes"]:
            out.append(f"- Note: {note}")
        out.append("")
    return "\n".join(out)


def nonpii_excludes(label: str) -> List[str]:
    """Hard-negative examples for `label` whose correct class is non_pii."""
    return [n for n, route in TAXONOMY[label]["exclude"] if route == NON_PII]


def routed_excludes(label: str) -> Dict[str, str]:
    return {n: route for n, route in TAXONOMY[label]["exclude"]}
