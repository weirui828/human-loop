"""
app.py — Human-in-the-Loop Streamlit Dashboard

Tabs:
  1. Live Triage Simulator: Enter a customer message and see the escalation prediction,
     from either the TF-IDF baseline or the fine-tuned DistilBERT (selectable).
  2. Human Label Audit: Review LLM-generated labels from llm_labeled_5k.csv,
     confirm or correct them, and save corrections to human_audit_5k.csv.

Run:
    streamlit run app.py
"""
import os
import csv
import time
import pickle
import pandas as pd
import streamlit as st

# Configure Page Layout & Styling
st.set_page_config(
    page_title="Human-in-the-Loop: Escalation Classifier & Dual-LLM Audit",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    :root {
        /* ── PALETTE ─────────────────────────────────────────────── */
        --mauve-100: #e6ddde;   /* the source page ground */
        --mauve-200: #d8cccd;
        --mauve-300: #c3b3b5;
        --mauve-600: #6d585b;
        --mauve-700: #574548;
        --mauve-800: #443235;   /* the source body text */

        --olive-100: #e3e8da;
        --olive-200: #c8d2b7;
        --olive-500: #79875e;   /* Outdoorsy Microbrew cover */
        --olive-600: #64714d;
        --olive-700: #4f5a3d;

        --blush-100: #f7ecec;
        --blush-300: #dab7ba;   /* Classy Luxury cover */
        --blush-700: #7d4f55;
        --cobalt-100: #dbe3f4;
        --cobalt-500: #19449c;  /* SaaS Tech cover */
        --cobalt-700: #123274;
        --peach-100: #fdeee7;   /* Retro Warmth cover, lightened */
        --peach-800: #7d4620;
        --pine-100: #dceadf;
        --pine-700: #1a4b29;
        --ember-100: #f8dedb;
        --ember-500: #a8372e;
        --ember-700: #7f2922;

        /* ── SEMANTIC — what every rule below uses ───────────────── */
        --surface-base: var(--mauve-100);
        --surface-raised: #ffffff;
        --surface-sunken: var(--mauve-200);
        --fg-default: var(--mauve-800);
        --fg-muted: var(--mauve-700);
        --fg-subtle: var(--mauve-600);
        --border-subtle: var(--mauve-200);
        --border-default: var(--mauve-300);
        --border-control: var(--mauve-600);
        --accent-default: var(--olive-600);
        --accent-hover: var(--olive-700);
        --accent-fg: var(--olive-700);
        --accent-subtle: var(--olive-100);

        /* ── SHARED — radius, elevation, type ────────────────────── */
        --radius-md: 0.375rem;
        --radius-lg: 0.5rem;
        --radius-xl: 0.75rem;
        --shadow-sm: 0 1px 3px 0 rgb(68 50 53 / 0.09), 0 1px 2px -1px rgb(68 50 53 / 0.06);
        --shadow-md: 0 4px 8px -2px rgb(68 50 53 / 0.1), 0 2px 4px -2px rgb(68 50 53 / 0.05);
        --font-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
    }

    .stApp {
        background-color: var(--surface-base);
        color: var(--fg-default);
        font-family: var(--font-sans);
    }
    .block-container {
        max-width: 1280px;
        padding-top: 1rem;
    }
    header[data-testid="stHeader"] { display: none; }

    /* Numbers must align across rows — tabular figures everywhere numeric. */
    .metric-value, .result-confidence, table { font-variant-numeric: tabular-nums; }

    /* Header. The design system has no gradient text; a solid deep plum with
       the olive doing the accenting is truer to it and far more legible. */
    .main-header {
        font-size: 2.25rem;
        font-weight: 600;
        line-height: 1.2;
        letter-spacing: -0.01em;
        color: var(--fg-default);
        margin-bottom: 0.25rem;
    }
    .sub-header {
        font-size: 0.875rem;
        color: var(--fg-muted);
        margin-bottom: 1.5rem;
    }

    /* Surfaces stack: page → card → well, in three steps. */
    .content-card {
        background-color: var(--surface-raised);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background-color: var(--surface-raised);
        border: 1px solid var(--border-subtle);
        border-radius: var(--radius-lg);
        padding: 1rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 600;
        color: var(--fg-default);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: var(--fg-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* The two LLM verdicts are a categorical pair, so they take two distinct
       lookbook hues rather than two shades of one. */
    .llm-card-gemini {
        background-color: var(--cobalt-100);
        border: 1px solid var(--cobalt-500);
        border-radius: var(--radius-md);
        padding: 1rem;
        color: var(--cobalt-700);
    }
    .llm-card-anthropic {
        background-color: var(--blush-100);
        border: 1px solid var(--blush-300);
        border-radius: var(--radius-md);
        padding: 1rem;
        color: var(--blush-700);
    }

    /* Customer vs agent — also categorical, also two distinct hues. */
    .chat-bubble-customer {
        background-color: var(--accent-subtle);
        border-left: 3px solid var(--olive-500);
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 0.5rem;
        color: var(--fg-default);
    }
    .chat-bubble-agent {
        background-color: var(--surface-sunken);
        border-left: 3px solid var(--border-control);
        padding: 0.75rem 1rem;
        border-radius: var(--radius-md);
        margin-bottom: 0.5rem;
        color: var(--fg-default);
    }

    /* Buttons. Flat olive, no gradient — the design system fills solid. */
    div.stButton > button,
    div.stButton > button[kind="primary"] {
        background: var(--accent-default) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 0.5rem 1rem !important;
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        box-shadow: none !important;
        transition: background-color 0.15s !important;
    }
    div.stButton > button:hover {
        background: var(--accent-hover) !important;
        opacity: 1 !important;
    }
    /* One focus indicator for the whole app: an outline, so it follows the
       radius and survives High Contrast Mode. */
    div.stButton > button:focus-visible,
    div[data-baseweb="select"] > div:focus-within,
    .stTextArea textarea:focus {
        outline: 2px solid var(--accent-default) !important;
        outline-offset: 2px !important;
        box-shadow: none !important;
    }

    /* Result cards use the status semantics, not the accent. */
    .result-card-escalated {
        background-color: var(--ember-100);
        border: 1px solid var(--ember-500);
        border-radius: var(--radius-lg);
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .result-card-safe {
        background-color: var(--pine-100);
        border: 1px solid var(--pine-700);
        border-radius: var(--radius-lg);
        padding: 1rem 1.25rem;
        margin-top: 0.75rem;
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
    }
    .result-icon {
        font-size: 1.25rem;
        line-height: 1;
        flex-shrink: 0;
        margin-top: 0.15rem;
    }
    .result-title {
        font-weight: 600;
        font-size: 1rem;
        color: var(--fg-default);
        margin-bottom: 0.15rem;
    }
    .result-confidence {
        font-size: 0.8125rem;
        color: var(--fg-muted);
    }

    /* Form controls. border-control is the only border step that meets the
       3:1 non-text contrast minimum, so interactive edges use it. */
    div[data-baseweb="select"] > div,
    .stTextArea textarea,
    .stTextInput input {
        background-color: var(--surface-raised) !important;
        border: 1px solid var(--border-control) !important;
        border-radius: var(--radius-md) !important;
        color: var(--fg-default) !important;
        font-family: var(--font-sans) !important;
    }
    div[data-baseweb="popover"] ul {
        background-color: var(--surface-raised) !important;
        border: 1px solid var(--border-default) !important;
        border-radius: var(--radius-md) !important;
    }
    div[data-baseweb="popover"] li:hover { background-color: var(--accent-subtle) !important; }

    /* Tabs — flat, with the olive marking the selected one. */
    button[data-baseweb="tab"] {
        font-family: var(--font-sans) !important;
        font-weight: 500 !important;
        color: var(--fg-muted) !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] { color: var(--fg-default) !important; }
    div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {
        background-color: var(--accent-default) !important;
    }

    h1, h2, h3, h4, h5 { color: var(--fg-default) !important; font-family: var(--font-sans) !important; }
    code { background-color: var(--surface-sunken) !important; color: var(--fg-default) !important; }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LLM_LABELS_CSV = os.path.join(BASE_DIR, "data", "twcs", "llm_labeled_5k.csv")
AUDIT_CSV = os.path.join(BASE_DIR, "data", "twcs", "human_audit_5k.csv")
TFIDF_PATH = os.path.join(BASE_DIR, "models", "twcs", "tfidf", "tfidf_twcs.pkl")
DISTILBERT_DIR = os.path.join(BASE_DIR, "models", "twcs", "distilbert")

# Both models are trained in-domain on the same 80/20 split of llm_labeled_5k.csv, so
# their predictions are directly comparable. Reported test macro F1: TF-IDF 0.7327,
# DistilBERT 0.7862 (models/twcs/tfidf/twcs_metrics.json and
# models/twcs/distilbert/twcs_distilbert_metrics.json).
MODEL_CHOICES = {
    "TF-IDF + Logistic Regression": "tfidf",
    "DistilBERT (fine-tuned)": "distilbert",
}


@st.cache_resource
def load_tfidf():
    """Load the trained TF-IDF + Logistic Regression pipeline."""
    if not os.path.exists(TFIDF_PATH):
        return None
    with open(TFIDF_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_distilbert():
    """Load the fine-tuned DistilBERT classifier, or None if the weights are absent.

    torch/transformers are imported lazily so that selecting TF-IDF never pays the
    (multi-second) import cost, and so the app still starts on an install without them.
    Inference runs on CPU: a single query is ~8 ms there, well inside an interactive
    budget, and it avoids depending on an accelerator being present.
    """
    if not os.path.exists(os.path.join(DISTILBERT_DIR, "model.safetensors")):
        return None
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return None
    tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_DIR)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "torch": torch}


def predict_escalation(kind, text):
    """Returns (is_escalated, p_escalated) for the selected model, or None if unavailable."""
    if kind == "tfidf":
        pipeline = load_tfidf()
        if pipeline is None:
            return None
        p_escalated = float(pipeline.predict_proba([text])[0][1])
    else:
        bundle = load_distilbert()
        if bundle is None:
            return None
        torch = bundle["torch"]
        enc = bundle["tokenizer"](text, truncation=True, max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = bundle["model"](**enc).logits
        p_escalated = float(torch.softmax(logits.float(), dim=-1)[0, 1])
    return p_escalated >= 0.5, p_escalated

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
st.markdown('<div class="sub-header">Cross-domain customer support evaluation & interactive human audit dashboard</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["🤖 Live Triage Simulator", "👥 Human Label Audit"])

# ==============================================================================
# TAB 1: LIVE TRIAGE SIMULATOR
# ==============================================================================
with tab1:
    st.subheader("Interactive Query Classifier")
    st.markdown("Enter a customer support query or thread to evaluate whether it requires human escalation or can be handled by self-service automation.")

    model_label = st.radio(
        "Classifier:",
        list(MODEL_CHOICES.keys()),
        horizontal=True,
        help=(
            "Both are trained in-domain on the same 80/20 split of the labeled TWCS "
            "sample. Test macro F1: TF-IDF 0.7327, DistilBERT 0.7862."
        ),
    )
    model_kind = MODEL_CHOICES[model_label]

    if model_kind == "distilbert" and load_distilbert() is None:
        st.info(
            "DistilBERT weights are not present (they are gitignored — ~256 MB). "
            "Run **notebook 04** to regenerate them into `models/twcs/distilbert/`, "
            "or select TF-IDF above."
        )

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
        started = time.perf_counter()
        result = predict_escalation(model_kind, user_input)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if result is None:
            if model_kind == "tfidf":
                st.warning(f"Model not found at `{TFIDF_PATH}`. Train it via notebook 02 first.")
            else:
                st.warning(
                    f"DistilBERT not available at `{DISTILBERT_DIR}`. Run notebook 04 to "
                    "produce the weights, or switch to TF-IDF."
                )
        else:
            is_escalated, p_escalated = result
            confidence = p_escalated if is_escalated else 1.0 - p_escalated
            if is_escalated:
                st.markdown(f"""
                <div class="result-card-escalated">
                    <div class="result-icon">🚨</div>
                    <div>
                        <div class="result-title">Escalation Required — Class 1 (Human Agent Needed)</div>
                        <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card-safe">
                    <div class="result-icon">🤖</div>
                    <div>
                        <div class="result-title">Bot Eligible — Class 0 (Self-Service / FAQ)</div>
                        <div class="result-confidence">Confidence: {confidence*100:.1f}%</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.caption(
                f"{model_label} · P(escalate) = {p_escalated:.3f} · {elapsed_ms:.1f} ms "
                "(first DistilBERT call includes model load)"
            )

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

            # The selectbox's own key is the single source of truth for the
            # current thread. Reset it if it points outside the current view
            # (e.g. the filter changed and dropped the previously-selected id).
            if (
                "audit_thread_select" not in st.session_state
                or st.session_state["audit_thread_select"] not in thread_ids
            ):
                st.session_state["audit_thread_select"] = thread_ids[0]
            idx = thread_ids.index(st.session_state["audit_thread_select"])

            # Prev/Next mutate the selectbox key directly. on_click callbacks run
            # before the rerun, so the selectbox picks up the new value instead of
            # its stale stored one (a plain `index=` arg would be ignored here).
            def _go_prev():
                cur = thread_ids.index(st.session_state["audit_thread_select"])
                st.session_state["audit_thread_select"] = thread_ids[max(0, cur - 1)]

            def _go_next():
                cur = thread_ids.index(st.session_state["audit_thread_select"])
                st.session_state["audit_thread_select"] = thread_ids[min(len(thread_ids) - 1, cur + 1)]

            nav_prev, nav_select, nav_next = st.columns([1, 6, 1])
            with nav_prev:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("⬅", use_container_width=True, key="nav_prev",
                          disabled=(idx == 0), on_click=_go_prev)
            with nav_select:
                selected_tid = st.selectbox(
                    "Select Thread to Audit:",
                    thread_ids,
                    key="audit_thread_select"
                )
            with nav_next:
                st.markdown("<br>", unsafe_allow_html=True)
                st.button("➡", use_container_width=True, key="nav_next",
                          disabled=(idx >= len(thread_ids) - 1), on_click=_go_next)
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
                # Status semantics, not raw hexes — these follow the theme.
                if llm_esc == 1:
                    verdict_html = '<div class="metric-value" style="color: var(--ember-700); font-size: 1.4rem;">🚨 Escalated (1)</div>'
                else:
                    verdict_html = '<div class="metric-value" style="color: var(--pine-700); font-size: 1.4rem;">🤖 Self-Service (0)</div>'
                st.markdown(
                    f'<div class="llm-card-gemini">'
                    f'{verdict_html}'
                    f'<div class="metric-label">Category: <code>{llm_cat}</code></div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

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
                    key=f"human_esc_radio_{tid_str}"
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
                    key=f"human_cat_select_{tid_str}"
                )

                human_notes = st.text_input(
                    "Notes (optional):",
                    value=default_notes,
                    placeholder="e.g. 'LLM missed the refund demand'",
                    key=f"human_notes_input_{tid_str}"
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
