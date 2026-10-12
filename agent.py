import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from profiler import profile

MODEL = "gemini-3.8-flash"

ALLOWED_ACTIONS = {
    "drop_duplicate_rows": "Remove exact duplicate rows",
    "convert_to_number": "Strip symbols like $ and , then convert to a number",
    "standardize_category": "Replace case/space variants with the most common form",
    "fill_with_median": "Fill missing numbers with the column median",
    "fill_with_most_common": "Fill missing text with the most common value",
    "set_to_missing": "Replace bad values with empty, for later review",
    "make_positive": "Turn negative values into positive",
    "drop_rows": "Remove the affected rows",
    "flag_only": "Do not change data; flag for human review",
}

SYSTEM = f"""You are DataGuard, a careful data quality analyst.
Steps:
1. Call get_profile to understand the dataset.
2. Call get_issue_summary to see the detected problems.
3. For each issue type, call get_issue_examples to look at real values.
4. For each issue type and column, call propose_fix once with ONE allowed action.
5. Finish with a short report (max 200 words) for a non-technical manager:
   what is wrong, why it matters, and what you propose. Use numbers.
Never invent issues the tools did not report.
Allowed actions: {json.dumps(ALLOWED_ACTIONS)}"""


def get_api_key():
    load_dotenv("key.env")
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        try:
            import streamlit as st
            key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            key = None
    return key


def run_agent(df: pd.DataFrame, issues: pd.DataFrame):
    """Returns (report_text, proposed_fixes_table, tool_call_log)."""
    proposals = {}
    log = []

    def get_profile() -> str:
        """Get dataset size, missing cells, duplicates and per-column statistics."""
        log.append("get_profile()")
        p = profile(df)
        return json.dumps({
            "rows": p["rows"], "columns": p["columns"],
            "duplicate_rows": p["duplicate_rows"], "missing_cells": p["missing_cells"],
            "column_profile": p["column_profile"].astype(str).to_dict("records"),
        })

    def get_issue_summary() -> str:
        """Count detected issues by type and column."""
        log.append("get_issue_summary()")
        if issues.empty:
            return "No issues detected."
        s = issues.fillna({"column": ""}).groupby(["type", "column"]).size()
        return s.reset_index(name="count").to_json(orient="records")

    def get_issue_examples(issue_type: str, limit: int = 5) -> str:
        """Show example issues of one type, including the actual values."""
        log.append(f"get_issue_examples('{issue_type}')")
        ex = issues[issues["type"] == issue_type].head(min(int(limit), 10))
        return ex.astype(str).to_json(orient="records")

    def propose_fix(issue_type: str, column: str, action: str, reason: str) -> str:
        """Propose one fix for one issue type in one column.
        Use column='' for row-level issues like duplicates or anomalies."""
        log.append(f"propose_fix('{issue_type}', '{column}', '{action}')")
        if action not in ALLOWED_ACTIONS:
            return f"Rejected. Use one of: {list(ALLOWED_ACTIONS)}"
        mask = issues["type"] == issue_type
        if column:
            mask &= issues["column"] == column
        proposals[(issue_type, column)] = {
            "issue_type": issue_type, "column": column or "(whole row)",
            "action": action, "affected": int(mask.sum()), "reason": reason,
        }
        return "Fix recorded."

    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[get_profile, get_issue_summary, get_issue_examples, propose_fix],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=30),
            temperature=0.2,
        ),
    )

    for attempt in range(4):
        try:
            response = chat.send_message("Audit this dataset and propose fixes.")
            break
        except errors.ServerError:
            if attempt == 3:
                raise
            time.sleep(10 * (attempt + 1))  # wait 10s, 20s, 30s

    report = response.text or "The agent finished without a written report."
    return report, pd.DataFrame(list(proposals.values())), log