import json

import pandas as pd
import plotly.express as px
import streamlit as st

from agent import run_agent
from benchmark import evaluate
from checks import run_all_checks
from fixes import apply_fixes, quality_score
from injector import inject_errors, make_clean_data
from profiler import profile

SAMPLE = "sample_dirty.csv"

st.set_page_config(page_title="DataGuard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ DataGuard AI")
st.caption("An AI agent that audits, explains and fixes data quality problems — "
           "with human approval and a measured accuracy benchmark.")


@st.cache_data
def get_issues(data: pd.DataFrame) -> pd.DataFrame:
    return run_all_checks(data)


def load_saved_run():
    with open("demo_run.json") as f:
        saved = json.load(f)
    st.session_state.update(report=saved["report"], fixes=pd.DataFrame(saved["fixes"]),
                            log=saved["log"], saved_run=True)


# ---------- Load data ----------
with st.sidebar:
    st.header("Load data")
    source = st.radio("Data source", ["Sample: NL prices (with errors)", "Upload my CSV"])
    if source == "Upload my CSV":
        file = st.file_uploader("CSV file (max 5 MB)", type="csv")
        if file is None:
            st.info("Upload a CSV to begin.")
            st.stop()
        if file.size > 5_000_000:
            st.error("File is larger than 5 MB.")
            st.stop()
        df = pd.read_csv(file)
        name = file.name
    else:
        df = pd.read_csv(SAMPLE)
        name = SAMPLE
    st.write(f"**{name}** — {len(df)} rows, {df.shape[1]} columns")

data_key = f"{name}-{df.shape}"
if st.session_state.get("data_key") != data_key:
    for k in ["report", "fixes", "log", "clean", "audit", "approved", "saved_run"]:
        st.session_state.pop(k, None)
    st.session_state["data_key"] = data_key

issues = get_issues(df)
score_before = quality_score(df, issues)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Overview", "🔍 Issues", "🤖 AI Agent", "✅ Results", "🎯 Benchmark"])

# ---------- Overview ----------
with tab1:
    p = profile(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", p["rows"])
    c2.metric("Duplicate rows", p["duplicate_rows"])
    c3.metric("Missing cells", p["missing_cells"])
    c4.metric("Quality score", f"{score_before}/100")
    st.subheader("Column profile")
    st.dataframe(p["column_profile"], use_container_width=True)
    st.subheader("Data preview")
    st.dataframe(df.head(20), use_container_width=True)

# ---------- Issues ----------
with tab2:
    if issues.empty:
        st.success("No issues found.")
    else:
        counts = issues["type"].value_counts().reset_index()
        counts.columns = ["type", "count"]
        st.plotly_chart(px.bar(counts, x="type", y="count", title="Issues by type"),
                        use_container_width=True)
        choice = st.selectbox("Filter by type", ["all"] + counts["type"].tolist())
        shown = issues if choice == "all" else issues[issues["type"] == choice]
        st.dataframe(shown.astype(str), use_container_width=True)

# ---------- AI Agent ----------
with tab3:
    st.write("The agent inspects the data with tools, then proposes fixes from a "
             "safe list. **It never changes data without your approval.**")
    runs = st.session_state.get("runs", 0)
    b1, b2 = st.columns([1, 3])
    run_live = b1.button("Run AI agent (live)", type="primary", disabled=runs >= 3)
    if name == SAMPLE and b2.button("Show saved agent run (no AI quota used)"):
        load_saved_run()

    if run_live:
        with st.spinner("Agent is auditing the data (up to 1–2 minutes)..."):
            try:
                report, fixes, log = run_agent(df, issues)
                st.session_state.update(report=report, fixes=fixes, log=log,
                                        runs=runs + 1, saved_run=False)
            except Exception as e:
                quota = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
                if quota and name == SAMPLE:
                    load_saved_run()
                    st.warning("The free AI quota is used up for today, so this shows "
                               "a saved run of the agent on the same sample data.")
                elif quota:
                    st.error("The free AI quota is used up for today. Please try tomorrow.")
                else:
                    st.error("The AI service is busy. Please try again in a minute.")
    if runs >= 3:
        st.caption("Demo limit reached (3 live runs per session).")

    if "report" in st.session_state:
        if st.session_state.get("saved_run"):
            st.info("Showing a saved agent run on the sample data.")
        st.subheader("Agent report")
        st.markdown(st.session_state["report"])
        with st.expander(f"Agent steps ({len(st.session_state['log'])} tool calls)"):
            for i, step in enumerate(st.session_state["log"], 1):
                st.code(f"{i}. {step}")
        st.subheader("Proposed fixes — approve or reject")
        fixes = st.session_state["fixes"].copy()
        if not fixes.empty:
            if "approve" not in fixes.columns:
                fixes.insert(0, "approve", fixes["action"] != "flag_only")
            edited = st.data_editor(
                fixes, use_container_width=True, hide_index=True,
                disabled=[c for c in fixes.columns if c != "approve"],
                column_config={"approve": st.column_config.CheckboxColumn("Approve")})
            st.session_state["approved"] = edited[edited["approve"]]

# ---------- Results ----------
with tab4:
    approved = st.session_state.get("approved")
    if approved is None or approved.empty:
        st.info("Run the AI agent and approve some fixes first.")
    else:
        if st.button("Apply approved fixes", type="primary"):
            clean, audit = apply_fixes(df, issues, approved)
            st.session_state.update(clean=clean, audit=audit)
        if "clean" in st.session_state:
            clean = st.session_state["clean"]
            audit = st.session_state["audit"]
            issues_after = run_all_checks(clean)
            score_after = quality_score(clean, issues_after)

            c1, c2, c3 = st.columns(3)
            c1.metric("Quality before", f"{score_before}/100")
            c2.metric("Quality after", f"{score_after}/100",
                      delta=round(score_after - score_before, 1))
            c3.metric("Changes made", len(audit))

            compare = pd.concat([
                issues["type"].value_counts().rename("before"),
                issues_after["type"].value_counts().rename("after")], axis=1).fillna(0)
            compare = compare.reset_index().melt(id_vars="type", var_name="stage",
                                                 value_name="count")
            st.plotly_chart(px.bar(compare, x="type", y="count", color="stage",
                                   barmode="group", title="Issues before vs after"),
                            use_container_width=True)

            st.subheader("Audit log (every change)")
            st.dataframe(audit, use_container_width=True)

            report_md = (f"# DataGuard AI report: {name}\n\n"
                         f"Quality score: {score_before} → {score_after}\n\n"
                         f"{st.session_state.get('report', '')}\n\n"
                         f"## Approved fixes\n\n{approved.to_markdown(index=False)}\n")
            d1, d2, d3 = st.columns(3)
            d1.download_button("Download clean CSV", clean.to_csv(index=False),
                               "clean_data.csv", "text/csv")
            d2.download_button("Download audit log", audit.to_csv(index=False),
                               "audit_log.csv", "text/csv")
            d3.download_button("Download report", report_md, "report.md", "text/markdown")

# ---------- Benchmark ----------
with tab5:
    st.write("We inject **known** errors into clean data, run the checks, "
             "and measure how many were caught.")
    rate = st.slider("Error rate per type", 0.005, 0.05, 0.02, 0.005)
    seed = st.number_input("Random seed", value=7, step=1)
    if st.button("Run benchmark"):
        dirty_b, truth = inject_errors(make_clean_data(), rate=rate, seed=int(seed))
        table, summary = evaluate(run_all_checks(dirty_b), truth)
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{summary['precision']:.1%}")
        c2.metric("Recall", f"{summary['recall']:.1%}")
        c3.metric("ML rows correct",
                  f"{summary['ml_rows_with_real_errors']}/{summary['ml_flagged_rows']}")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(table, x="error_type", y=["precision", "recall"],
                               barmode="group", title="Accuracy by error type"),
                        use_container_width=True)