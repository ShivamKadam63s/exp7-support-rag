"""
streamlit_app.py  -  Customer Support RAG, deployed directly with Streamlit.

Tab 1 "Ask"         : the customer-facing question box (validation + safe errors).
Tab 2 "Monitoring"  : live counters, latency, recent events and a real backend check.

Run locally :  streamlit run streamlit_app.py
"""
import logging
import os
import threading
import time
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Support RAG", page_icon="💬", layout="wide")

# ----------------------------------------------------------------------
# SECRETS -> ENVIRONMENT VARIABLES
# Streamlit keeps secrets in st.secrets (Cloud: Settings -> Secrets; local:
# .streamlit/secrets.toml). support_rag.py reads environment variables, so we
# copy them across BEFORE importing it. If there are no secrets, we just go on
# (a local .env file is then read by support_rag.py).
# ----------------------------------------------------------------------
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, str):
            os.environ.setdefault(_key, _value)
except Exception:
    pass

# Log lines appear in "Manage app -> Logs" on Streamlit Cloud.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("streamlit_app")

# Importing builds the TF-IDF index once per server process (Python caches the module),
# so it is NOT rebuilt on every click.
import support_rag
from support_rag import FALLBACK_MESSAGE, LLMBackendError, ask_support, check_backend

MAX_QUESTION_LENGTH = 1000
MAX_QUESTIONS_PER_SESSION = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "30"))
APP_PASSCODE = os.getenv("APP_PASSCODE", "")

EXAMPLES = [
    "How can I track my shipment?",
    "Can I return an unused item after 10 days?",
    "When should I expect a refund after an approved return is received?",
    "I cannot sign in to my account. What should I do?",
    "My payment seems to have been charged twice. What should I do?",
    "The website is not loading. What troubleshooting steps can I try?",
    "What is the live status of order AB12345?",
    "Can you tell me the password for my account?",
    "Do you sell home insurance?",
    "What is the capital of France?",
    "Can you recommend a good restaurant nearby?",
    "What is the price of the premium membership?",
]


# ----------------------------------------------------------------------
# MONITORING STORE
# cache_resource = ONE shared object for all visitors of this server process.
# It lives in memory only: it resets when the app sleeps, reboots or redeploys.
# ----------------------------------------------------------------------
@st.cache_resource
def get_store():
    return {
        "lock": threading.Lock(),
        "started": time.time(),
        "events": deque(maxlen=500),   # newest events; old ones drop off
        "totals": {"requests": 0, "answered": 0, "answered_after_retry": 0,
                   "fallback": 0, "errors": 0, "rejected": 0},
    }


def record(outcome, latency_ms=None, intent="", retries=0, question_chars=0):
    """Add one event. The question TEXT is never stored (may hold personal data)."""
    store = get_store()
    with store["lock"]:
        store["totals"]["requests"] += 1
        store["totals"][outcome] += 1
        store["events"].append({
            "time_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            "outcome": outcome,
            "latency_ms": latency_ms,
            "intent": intent,
            "retries": retries,
            "question_chars": question_chars,
        })


# ----------------------------------------------------------------------
# OPTIONAL PASSCODE GATE (stops strangers using up the free API quota)
# ----------------------------------------------------------------------
if APP_PASSCODE and not st.session_state.get("authorised"):
    st.title("Customer Support Assistant")
    entered = st.text_input("Enter the passcode to continue", type="password")
    if entered:
        if entered == APP_PASSCODE:
            st.session_state["authorised"] = True
            st.rerun()
        else:
            st.error("Incorrect passcode.")
    st.stop()

# ----------------------------------------------------------------------
# SIDEBAR: configuration summary (never shows the key itself)
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("About this app")
    st.write("Answers customer questions using only a small FAQ. "
             "If the FAQ does not cover a question, it says so.")
    st.caption(f"Model: {support_rag.LLM_MODEL}")
    st.caption("Retrieval: TF-IDF over FAQ chunks")
    st.caption(f"FAQ chunks indexed: {len(support_rag.chunks)}")
    st.caption("API key configured: " + ("yes" if support_rag.LLM_API_KEY else "NO"))

st.title("💬 Customer Support Assistant")
tab_ask, tab_monitor = st.tabs(["Ask", "Monitoring"])

# ======================================================================
# TAB 1: ASK
# ======================================================================
with tab_ask:
    st.write("Type a question, or pick one of the example questions.")
    CUSTOM = "(type my own question)"
    example = st.selectbox("Example questions", [CUSTOM] + EXAMPLES)
    default_text = "" if example == CUSTOM else example
    # Changing the key when the example changes resets the text box content.
    question = st.text_area("Your question", value=default_text,
                            max_chars=MAX_QUESTION_LENGTH, key=f"question_{example}")

    if st.button("Ask", type="primary"):
        q = question.strip()
        used = st.session_state.get("asked", 0)

        # --- input validation (same rules as the FastAPI version) ---
        if not q:
            record("rejected", question_chars=0)
            st.warning("Please type a question first.")
        elif len(q) > MAX_QUESTION_LENGTH:
            record("rejected", question_chars=len(q))
            st.warning(f"Question is too long. Please keep it under {MAX_QUESTION_LENGTH} characters.")
        elif used >= MAX_QUESTIONS_PER_SESSION:
            st.warning("Question limit reached for this session. Please reload the page.")
        else:
            st.session_state["asked"] = used + 1
            start = time.perf_counter()
            try:
                with st.spinner("Searching the FAQ and writing an answer..."):
                    result = ask_support(q)
            except LLMBackendError as e:
                latency = (time.perf_counter() - start) * 1000
                logger.error("ask FAILED | reason=llm_backend (%s) | question_chars=%d | latency_ms=%.1f",
                             e, len(q), latency)
                record("errors", latency_ms=round(latency, 1), question_chars=len(q))
                st.error("The answer service is temporarily unavailable. Please try again later.")
            except Exception as e:
                latency = (time.perf_counter() - start) * 1000
                logger.error("ask FAILED | reason=unexpected (%s) | latency_ms=%.1f",
                             type(e).__name__, latency)
                record("errors", latency_ms=round(latency, 1), question_chars=len(q))
                st.error("Something went wrong. Please try again later.")
            else:
                latency = round((time.perf_counter() - start) * 1000, 1)
                answer = result.get("final_answer", FALLBACK_MESSAGE)
                answered = answer != FALLBACK_MESSAGE
                retries = result.get("retry_count", 0)
                intent = result.get("intent", "unknown")
                if not answered:
                    outcome = "fallback"
                elif retries > 0:
                    outcome = "answered_after_retry"
                else:
                    outcome = "answered"
                logger.info("ask OK | outcome=%s | intent=%s | retries=%s | question_chars=%d | latency_ms=%.1f",
                            outcome, intent, retries, len(q), latency)
                record(outcome, latency_ms=latency, intent=intent,
                       retries=retries, question_chars=len(q))

                if answered:
                    st.success(answer)
                else:
                    st.info(answer)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Result", "Answered" if answered else "Fallback")
                c2.metric("Intent", intent)
                c3.metric("Retries", retries)
                c4.metric("Time (ms)", latency)
                with st.expander("Sources used"):
                    st.write(", ".join(result.get("sources", [])) or "None")
                    for passage in result.get("passages", []):
                        st.caption(passage)

# ======================================================================
# TAB 2: MONITORING
# ======================================================================
with tab_monitor:
    store = get_store()
    with store["lock"]:
        totals = dict(store["totals"])
        events = list(store["events"])
    uptime_min = (time.time() - store["started"]) / 60

    st.subheader("Service health")
    h1, h2, h3 = st.columns(3)
    h1.metric("App process running for", f"{uptime_min:.0f} min")
    h2.metric("FAQ index ready", "yes" if support_rag.index_ready() else "NO")
    h3.metric("API key configured", "yes" if support_rag.LLM_API_KEY else "NO")

    # A real check of the provider. Costs one tiny request, so it only runs on click.
    if st.button("Run live backend check"):
        with st.spinner("Contacting the model provider..."):
            ok, detail = check_backend()
        (st.success if ok else st.error)(("OK: " if ok else "FAILED: ") + detail)
    st.caption("This tests the provider, key and model name. A basic health endpoint cannot do that.")

    st.subheader("Traffic since the app process started")
    t = totals
    handled = t["answered"] + t["answered_after_retry"] + t["fallback"] + t["errors"]
    success_pct = 100 * (t["answered"] + t["answered_after_retry"] + t["fallback"]) / handled if handled else 0
    df = pd.DataFrame(events)
    lat = df["latency_ms"].dropna() if not df.empty else pd.Series(dtype=float)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Questions received", t["requests"])
    m2.metric("Answered from FAQ", t["answered"] + t["answered_after_retry"])
    m3.metric("Fallback (not in FAQ)", t["fallback"])
    m4.metric("Backend errors", t["errors"])
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Rejected inputs", t["rejected"])
    m6.metric("Answered after retry", t["answered_after_retry"])
    m7.metric("Average latency (ms)", f"{lat.mean():.0f}" if len(lat) else "-")
    m8.metric("95th percentile (ms)", f"{lat.quantile(0.95):.0f}" if len(lat) else "-")
    st.caption(f"Requests that did not hit a backend error: {success_pct:.0f}%")

    if len(lat):
        st.subheader("Latency of recent requests (ms)")
        st.line_chart(lat.reset_index(drop=True))
        st.subheader("Recent events (newest first)")
        st.dataframe(df.iloc[::-1].head(50), hide_index=True)
    else:
        st.info("No requests yet. Ask a question in the Ask tab, then come back here.")

    st.warning("These numbers live in memory. They reset when the app sleeps, restarts or is "
               "redeployed. For history, check Manage app -> Logs on Streamlit Cloud.")
