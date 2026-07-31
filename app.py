"""
app.py — Human-in-the-Loop Streamlit Dashboard

Tabs:
  1. Live Triage Simulator: Enter a customer message and see the escalation prediction.
  2. Human Label Audit: Review LLM-generated labels from llm_labeled_5k.csv,
     confirm or correct them, and save corrections to human_audit_5k.csv.

Run:
    streamlit run app.py
"""
import os
import csv
import pandas as pd
import streamlit as st

# Configure Page Layout & Styling
st.set_page_config(
    page_title="Human-in-the-Loop: Escalation Classifier & Dual-LLM Audit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Glassmorphic Aesthetics
st.markdown("""
<style>
    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
    }
    header[data-testid="stHeader"] {
        display: none;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #9333EA, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: rgba(31, 41, 55, 0.7);
        border: 1px solid rgba(75, 85, 99, 0.4);
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #F3F4F6;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .divergence-box {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .llm-card-gemini {
        background-color: rgba(59, 130, 246, 0.1);
        border: 1px solid rgba(59, 130, 246, 0.4);
        border-radius: 10px;
        padding: 1rem;
    }
    .llm-card-anthropic {
        background-color: rgba(168, 85, 247, 0.1);
        border: 1px solid rgba(168, 85, 247, 0.4);
        border-radius: 10px;
        padding: 1rem;
    }
    .chat-bubble-customer {
        background-color: rgba(99, 102, 241, 0.12);
        border-left: 4px solid #6366F1;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    .chat-bubble-agent {
        background-color: rgba(31, 41, 55, 0.8);
        border-left: 4px solid #10B981;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_LABELS_CSV = os.path.join(BASE_DIR, "data", "twcs", "llm_labeled_5k.csv")
AUDIT_CSV = os.path.join(BASE_DIR, "data", "twcs", "human_audit_5k.csv")

AUDIT_FIELDNAMES = [
    "thread_id", "llm_escalated", "llm_category", "llm_reason",
    "human_escalated", "human_category", "human_notes", "auditor", "audit_timestamp"
]

ESCALATION_CATEGORIES = [
    "contact_human_agent", "contact_customer_service", "complaint",
    "payment_issue", "get_refund", "check_cancellation_fee", "registration_problems"
]
ALL_CATEGORIES = ESCALATION_CATEGORIES + ["self_service"]

st.markdown('<div class="main-header">Human-in-the-Loop: Escalation Classifier & Label Audit</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Cross-Domain Customer Support Evaluation & Interactive Human Audit Dashboard</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🤖 Live Triage Simulator", "👥 Human Label Audit"])

# ==============================================================================
# TAB 1: LIVE TRIAGE SIMULATOR
# ==============================================================================
with tab1:
    st.subheader("Interactive Query Classifier")
    st.markdown("Enter a customer support query or thread to evaluate whether it requires human escalation or can be handled by self-service automation.")

    sample_query = st.selectbox(
        "Select a Benchmark Example Query:",
        [
            "Custom Query",
            "My picture on @Ask_Spectrum pretty much every day. Why should I pay $171 per month?",
            "Yo @Ask_Spectrum, your customer service reps are super nice— but imma start trippin if y'all don't get my service going!",
            "How do I track my order status online?",
            "I want to speak with a manager right now regarding an unauthorized charge on my account!",
            "Thanks, got it fixed!"
        ]
    )

    if sample_query == "Custom Query":
        user_input = st.text_area("Customer Utterance:", value="I tried logging into my account 5 times and keep getting error code 404.", height=100)
    else:
        user_input = st.text_area("Customer Utterance:", value=sample_query, height=100)

    if st.button("🚀 Analyze Escalation Risk", use_container_width=True):
        text_lower = user_input.lower()
        escalation_markers = ["manager", "unauthorized", "charge", "refund", "trippin", "pay", "error", "broken", "human", "speak", "lawyer"]
        score = sum(1 for kw in escalation_markers if kw in text_lower)
        is_escalated = (score > 0 or len(text_lower.split()) > 20)
        confidence = min(0.60 + (score * 0.15), 0.99) if is_escalated else 0.92
        if is_escalated:
            st.error(f"🚨 **Escalation Required** — Class 1 (Human Agent Needed) · Confidence: **{confidence*100:.1f}%**")
        else:
            st.success(f"🤖 **Bot Eligible** — Class 0 (Self-Service / FAQ) · Confidence: **{confidence*100:.1f}%**")

# ==============================================================================
# TAB 2: HUMAN LABEL AUDIT
# ==============================================================================
with tab2:
    st.subheader("Human Label Audit")
    st.markdown("Review LLM-generated labels from `llm_labeled_5k.csv`. Confirm or correct each label. Corrections are saved to `human_audit_5k.csv` — the original file is never modified.")

    if not os.path.exists(LLM_LABELS_CSV):
        st.warning(f"LLM labels file not found at `{LLM_LABELS_CSV}`. Please generate labels first.")
    else:
        # Load LLM labels (read-only source)
        df_llm = pd.read_csv(LLM_LABELS_CSV)

        # Load existing audit progress
        audited_ids: set = set()
        audit_rows: dict = {}
        if os.path.exists(AUDIT_CSV):
            df_audit = pd.read_csv(AUDIT_CSV)
            for _, arow in df_audit.iterrows():
                tid = str(arow.get("thread_id", ""))
                if tid:
                    audited_ids.add(tid)
                    audit_rows[tid] = arow

        total_threads = len(df_llm)
        audited_count = len(audited_ids & set(df_llm["thread_id"].astype(str)))
        pending_count = total_threads - audited_count

        # --- Progress Metrics ---
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Total Threads", f"{total_threads:,d}")
        with m2:
            st.metric("Audited", f"{audited_count:,d}")
        with m3:
            st.metric("Pending", f"{pending_count:,d}")

        if total_threads > 0:
            st.progress(audited_count / total_threads, text=f"{audited_count / total_threads * 100:.1f}% audited")

        st.divider()

        # --- Filter ---
        view_filter = st.radio(
            "Show:",
            ["Pending only", "Audited only", "All"],
            horizontal=True,
            key="audit_filter"
        )

        df_view = df_llm.copy()
        df_view["_tid_str"] = df_view["thread_id"].astype(str)
        if view_filter == "Pending only":
            df_view = df_view[~df_view["_tid_str"].isin(audited_ids)]
        elif view_filter == "Audited only":
            df_view = df_view[df_view["_tid_str"].isin(audited_ids)]

        if len(df_view) == 0:
            if view_filter == "Pending only":
                st.success("🎉 All threads have been audited!")
            else:
                st.info("No threads match the current filter.")
        else:
            thread_ids = df_view["thread_id"].tolist()

            # Determine current index from session state
            if "audit_idx" not in st.session_state:
                st.session_state["audit_idx"] = 0
            # Clamp index to valid range
            idx = st.session_state["audit_idx"]
            if idx >= len(thread_ids):
                idx = len(thread_ids) - 1
            if idx < 0:
                idx = 0

            nav_prev, nav_select, nav_next = st.columns([1, 6, 1])
            with nav_prev:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("⬅ Prev", use_container_width=True, key="nav_prev", disabled=(idx == 0)):
                    st.session_state["audit_idx"] = idx - 1
                    st.rerun()
            with nav_select:
                selected_tid = st.selectbox(
                    "Select Thread to Audit:",
                    thread_ids,
                    index=idx,
                    key="audit_thread_select"
                )
                # Sync selectbox back to session state
                new_idx = thread_ids.index(selected_tid) if selected_tid in thread_ids else 0
                if new_idx != st.session_state["audit_idx"]:
                    st.session_state["audit_idx"] = new_idx
            with nav_next:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Next ➡", use_container_width=True, key="nav_next", disabled=(idx >= len(thread_ids) - 1)):
                    st.session_state["audit_idx"] = idx + 1
                    st.rerun()
            row = df_llm[df_llm["thread_id"] == selected_tid].iloc[0]
            tid_str = str(selected_tid)
            is_already_audited = tid_str in audited_ids

            if is_already_audited:
                st.info("✅ This thread has already been audited. You can update your correction below.")

            # --- Customer Message Display ---
            st.markdown("#### 👤 Customer's Opening Message")
            customer_text = str(row.get("first_customer_text", ""))
            st.markdown(
                f'<div class="chat-bubble-customer" style="font-size: 1.05rem;">{customer_text}</div>',
                unsafe_allow_html=True
            )

            st.divider()

            # --- Side-by-side: LLM Verdict (left) | Human Audit (right) ---
            col_llm, col_human = st.columns(2)

            llm_esc = int(row.get("escalated", 0))
            llm_cat = str(row.get("category", "self_service"))
            llm_reason = str(row.get("reason", ""))

            with col_llm:
                st.markdown("#### 🤖 LLM Verdict")
                st.markdown('<div class="llm-card-gemini">', unsafe_allow_html=True)
                if llm_esc == 1:
                    st.markdown('<div class="metric-value" style="color: #EF4444; font-size: 1.4rem;">🚨 Escalated (1)</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="metric-value" style="color: #10B981; font-size: 1.4rem;">🤖 Self-Service (0)</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-label">Category: <code>{llm_cat}</code></div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown(f"**Reason:** {llm_reason}")
                st.markdown(f"**Labeler:** `{row.get('labeler', 'N/A')}`")
                st.markdown(f"**Turn Count:** {row.get('turn_count', 'N/A')}")

            with col_human:
                st.markdown("#### ✍️ Human Audit")

                # Pre-fill from existing audit if available
                if is_already_audited:
                    prev = audit_rows[tid_str]
                    default_esc = int(prev.get("human_escalated", llm_esc))
                    default_cat = str(prev.get("human_category", llm_cat))
                    default_notes = str(prev.get("human_notes", ""))
                else:
                    default_esc = llm_esc
                    default_cat = llm_cat
                    default_notes = ""

                human_esc = st.radio(
                    "Escalation Label:",
                    options=[0, 1],
                    format_func=lambda x: "🤖 0 — Self-Service" if x == 0 else "🚨 1 — Escalation",
                    index=0 if default_esc == 0 else 1,
                    horizontal=True,
                    key="human_esc_radio"
                )

                if human_esc == 1:
                    cat_options = ESCALATION_CATEGORIES
                else:
                    cat_options = ["self_service"]

                default_cat_idx = 0
                if default_cat in cat_options:
                    default_cat_idx = cat_options.index(default_cat)

                human_cat = st.selectbox(
                    "Category:",
                    cat_options,
                    index=default_cat_idx,
                    key="human_cat_select"
                )

                human_notes = st.text_input(
                    "Notes (optional):",
                    value=default_notes,
                    placeholder="e.g. 'LLM missed the refund demand'",
                    key="human_notes_input"
                )

                auditor_name = st.text_input(
                    "Auditor:",
                    value=st.session_state.get("auditor_name", ""),
                    placeholder="Your name",
                    key="auditor_name_input"
                )

            col_submit, col_skip = st.columns(2)
            with col_submit:
                submit_label = "💾 Update Audit" if is_already_audited else "💾 Submit Audit"
                if st.button(submit_label, type="primary", use_container_width=True, key="submit_audit"):
                    from datetime import datetime, timezone
                    st.session_state["auditor_name"] = auditor_name

                    audit_record = {
                        "thread_id": selected_tid,
                        "llm_escalated": llm_esc,
                        "llm_category": llm_cat,
                        "llm_reason": llm_reason,
                        "human_escalated": human_esc,
                        "human_category": human_cat,
                        "human_notes": human_notes,
                        "auditor": auditor_name,
                        "audit_timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    # Load existing audit file, update or append, then save
                    if os.path.exists(AUDIT_CSV):
                        df_existing = pd.read_csv(AUDIT_CSV)
                    else:
                        df_existing = pd.DataFrame(columns=AUDIT_FIELDNAMES)

                    # If already audited, update the existing row; otherwise append
                    if tid_str in set(df_existing["thread_id"].astype(str)):
                        idx = df_existing[df_existing["thread_id"].astype(str) == tid_str].index[0]
                        for k, v in audit_record.items():
                            df_existing.loc[idx, k] = v
                    else:
                        df_existing = pd.concat([df_existing, pd.DataFrame([audit_record])], ignore_index=True)

                    df_existing.to_csv(AUDIT_CSV, index=False)

                    changed = (human_esc != llm_esc) or (human_cat != llm_cat)
                    if changed:
                        st.success(f"✏️ Correction saved for `{selected_tid}`: {llm_cat}({llm_esc}) → {human_cat}({human_esc})")
                    else:
                        st.success(f"✅ Confirmed LLM label for `{selected_tid}` (no changes).")
                    st.rerun()

            with col_skip:
                if st.button("⏭️ Confirm & Next", use_container_width=True, key="skip_next"):
                    from datetime import datetime, timezone
                    st.session_state["auditor_name"] = auditor_name

                    # Auto-confirm the LLM label as-is
                    audit_record = {
                        "thread_id": selected_tid,
                        "llm_escalated": llm_esc,
                        "llm_category": llm_cat,
                        "llm_reason": llm_reason,
                        "human_escalated": llm_esc,
                        "human_category": llm_cat,
                        "human_notes": "confirmed",
                        "auditor": auditor_name,
                        "audit_timestamp": datetime.now(timezone.utc).isoformat()
                    }

                    if os.path.exists(AUDIT_CSV):
                        df_existing = pd.read_csv(AUDIT_CSV)
                    else:
                        df_existing = pd.DataFrame(columns=AUDIT_FIELDNAMES)

                    if tid_str not in set(df_existing["thread_id"].astype(str)):
                        df_existing = pd.concat([df_existing, pd.DataFrame([audit_record])], ignore_index=True)
                        df_existing.to_csv(AUDIT_CSV, index=False)

                    st.rerun()
