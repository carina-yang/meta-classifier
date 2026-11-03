"""
text_processing.py
------------------
Converts (table_name, column_name, data_type) into a normalized text string
for model training.

Normalizations applied:
- Lowercase conversion
- Letter/digit boundary splitting (addr1 -> "addr 1", tel_1 -> "tel 1")
- Schema prefix removal (dbo.Customer -> "customer")
- Separator normalization (dots/slashes/colons/underscores -> spaces)
- CamelCase splitting (DateOfBirth -> "date of birth")
- Data type normalization (VARCHAR(255) -> "varchar 255")
- Bracket/quote stripping

Output format: "column: <col> | table: <tbl> | type: <type>"
Missing values become "unknown" to provide explicit signal for uninformative contexts.
"""
import re

import pandas as pd

from config import COL_DATA_TYPE, COL_NAME, COL_TABLE_NAME

_WRAP = "[]\"'`"


def normalize_identifier(text: str) -> str:
    """snake/camel/kebab/dotted identifiers -> lowercase space-separated tokens.
        CUST_DOB    -> "cust dob"      DateOfBirth -> "date of birth"
        EmailAddr   -> "email addr"    addr_ln1    -> "addr ln 1"
        ABCDef      -> "abc def"       tel_1       -> "tel 1"
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    s = text.strip().strip(_WRAP)
    s = re.sub(r"[_\-./\\:]+", " ", s)
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)          # camelCase
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)        # ABCDef -> ABC Def
    s = re.sub(r"(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])", " ", s)  # letter/digit
    return re.sub(r"\s+", " ", s).strip().lower()


def normalize_table_name(text: str) -> str:
    """Drop schema/database prefixes (dbo.Customer -> customer) then normalize."""
    if not isinstance(text, str) or not text.strip():
        return ""
    last = re.split(r"\.", text.strip().strip(_WRAP))[-1]
    return normalize_identifier(last)


def normalize_data_type(text: str) -> str:
    """VARCHAR(255) -> 'varchar 255'; DECIMAL(10,2) -> 'decimal 10 2'."""
    if not isinstance(text, str) or not text.strip():
        return ""
    s = re.sub(r"[(),\[\]]+", " ", text.strip().lower())
    return re.sub(r"\s+", " ", s).strip()


def build_text(df: pd.DataFrame) -> pd.Series:
    """Build the all_text field from table_name, column_name, data_type."""
    col = df[COL_NAME].map(normalize_identifier).replace("", "unknown")
    tbl = df[COL_TABLE_NAME].map(normalize_table_name).replace("", "unknown")
    typ = df[COL_DATA_TYPE].map(normalize_data_type).replace("", "unknown")
    return ("column: " + col + " | table: " + tbl + " | type: " + typ)
