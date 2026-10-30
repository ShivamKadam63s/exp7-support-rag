"""
support_rag.py  -  RAG pipeline for the Streamlit deployment.

Same workflow as Experiments 6 and 7 (classify -> retrieve -> draft -> review ->
retry once -> fallback), with TWO deliberate differences:

  1. RETRIEVAL uses TF-IDF (scikit-learn) instead of sentence embeddings + FAISS.
     Reason: Streamlit Community Cloud has only about 1 GB of memory, and PyTorch
     (needed by the embedding model) does not fit comfortably. TF-IDF matches
     words, not meaning, so it is weaker on questions phrased with different words.
  2. A check_backend() function is added, so the app can really test the LLM
     provider (the /health idea from Experiment 7 could not do this).

The LLM is still a cloud API (Groq). Settings come from environment variables.
"""
import logging
import os
import re
from typing import List, TypedDict

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, START, END
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()   # local runs: reads .env. On Streamlit Cloud the secrets are already env vars.
logger = logging.getLogger("support_rag")

# ----------------------------------------------------------------------
# 1. BACKEND CONFIGURATION (environment variables, never hard-coded)
# ----------------------------------------------------------------------
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "30"))


class LLMBackendError(Exception):
    """Any provider problem (bad key, quota, network, wrong model name...)."""


_client = None


def get_client() -> OpenAI:
    """Create the API client once and reuse it."""
    global _client
    if not LLM_API_KEY:
        raise LLMBackendError("LLM_API_KEY is not configured")
    if _client is None:
        _client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, max_retries=1)
    return _client


def _llm_kwargs() -> dict:
    """Extra request settings. Only gpt-oss models accept 'reasoning_effort'."""
    extra = {}
    if "gpt-oss" in LLM_MODEL.lower():
        extra["reasoning_effort"] = "low"   # keep the hidden thinking short
    return extra


def check_backend():
    """
    Real provider check: sends a tiny request and reports (ok, detail).
    detail holds only an error TYPE, never a message that could contain the key.
    """
    try:
        get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": "Reply with the word OK."}],
            max_tokens=64,
            timeout=LLM_TIMEOUT_SECONDS,
            **_llm_kwargs(),
        )
        return True, "Provider reachable, key and model accepted"
    except LLMBackendError as e:
        return False, str(e)
    except OpenAIError as e:
        return False, f"{type(e).__name__} (status={getattr(e, 'status_code', 'n/a')})"


# ----------------------------------------------------------------------
# 2. WORD LIST + TOKENIZER (shared by retrieval and the coverage check)
# ----------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did",
    "can", "could", "will", "would", "should", "i", "you", "me", "my", "we", "our",
    "your", "it", "its", "of", "to", "in", "on", "at", "for", "from", "with", "about",
    "what", "when", "where", "how", "why", "who", "which", "this", "that", "there",
    "any", "and", "or", "not", "no", "if", "so", "after", "tell", "please", "have",
    "has", "had", "seems", "seem", "get", "gets", "try", "steps",
}


def tokenize(text: str) -> List[str]:
    """Lower-case words, stop words removed, cut to 5 letters so that
    'return/returns' and 'track/tracking' count as the same word."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:5] for w in words
            if len(w) > 2 and w not in STOPWORDS and w not in ENGLISH_STOP_WORDS]


# ----------------------------------------------------------------------
# 3. KNOWLEDGE BASE + TF-IDF RETRIEVER (built ONCE when the app process starts)
# ----------------------------------------------------------------------
class TfidfRetriever:
    """Tiny retriever with the same .invoke(query) idea as a LangChain retriever."""

    def __init__(self, docs: List[Document], k: int = 3):
        self.docs = docs
        self.k = k
        self.vectorizer = TfidfVectorizer(analyzer=tokenize, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform([d.page_content for d in docs])

    def invoke(self, query: str) -> List[Document]:
        scores = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = scores.argsort()[::-1][: self.k]
        return [self.docs[i] for i in order if scores[i] > 0]   # ignore chunks with no overlap


logger.info("Loading knowledge base...")
with open("support_faq.txt", "r", encoding="utf-8") as f:
    faq_text = f.read()

faq_doc = Document(page_content=faq_text, metadata={"source": "support_faq.txt"})
splitter = RecursiveCharacterTextSplitter(chunk_size=450, chunk_overlap=60)
chunks = splitter.split_documents([faq_doc])
retriever = TfidfRetriever(chunks, k=3)
logger.info("Index ready. Chunks indexed: %d", len(chunks))


def index_ready() -> bool:
    return len(chunks) > 0


# ----------------------------------------------------------------------
# 4. GRAPH STATE AND CONSTANTS
# ----------------------------------------------------------------------
class SupportState(TypedDict, total=False):
    question: str
    intent: str
    search_query: str
    context: str
    sources: List[str]
    passages: List[str]
    draft: str
    retry_count: int
    final_answer: str


INSUFFICIENT_MARKER = "INSUFFICIENT_CONTEXT"
FALLBACK_MESSAGE = (
    "I couldn't find a reliable answer in the available FAQ. "
    "Please contact the support team through the Help section. "
    "Do not share your password, one-time code, or full payment-card details."
)


# ----------------------------------------------------------------------
# 5. GRAPH NODES
# ----------------------------------------------------------------------
def classify_intent(state: SupportState):
    q = state["question"].lower()
    if any(w in q for w in ["refund", "return", "exchange"]):
        intent = "returns_refunds"
    elif any(w in q for w in ["track", "tracking", "shipment", "delivery", "order status"]):
        intent = "order_tracking"
    elif any(w in q for w in ["password", "sign in", "login", "account", "verification"]):
        intent = "account_access"
    elif any(w in q for w in ["charged", "payment", "billing", "invoice", "card"]):
        intent = "billing_payments"
    elif any(w in q for w in ["website", "error", "loading", "browser", "technical"]):
        intent = "technical_support"
    else:
        intent = "general_support"
    return {"intent": intent, "search_query": state["question"],
            "retry_count": state.get("retry_count", 0)}


def retrieve_context(state: SupportState):
    docs = retriever.invoke(state["search_query"])
    return {
        "context": "\n\n".join(d.page_content for d in docs),
        "sources": list(dict.fromkeys(d.metadata.get("source", "unknown") for d in docs)),
        "passages": [d.page_content for d in docs],
    }


def draft_answer(state: SupportState):
    prompt = f"""Use only the FAQ CONTEXT to answer the CUSTOMER QUESTION.
Do not invent order status, refund approvals, timelines, or company policies.
If the context does not contain the answer, output exactly:
{INSUFFICIENT_MARKER}
Do not ask for passwords, one-time codes, or full payment-card details.
Keep the answer clear and concise, in at most three sentences.

FAQ CONTEXT:
{state.get("context", "")}

CUSTOMER QUESTION:
{state["question"]}

ANSWER:"""
    messages = [
        {"role": "system", "content": "You are a polite customer-support assistant for an online store."},
        {"role": "user", "content": prompt},
    ]
    try:
        resp = get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=1024,                 # room for hidden reasoning + the answer
            timeout=LLM_TIMEOUT_SECONDS,
            **_llm_kwargs(),
        )
        text = (resp.choices[0].message.content or "").strip()
    except LLMBackendError:
        raise
    except OpenAIError as e:
        # Log only the error TYPE and status. Provider messages may echo part of the key.
        logger.error("LLM provider error: %s (status=%s)",
                     type(e).__name__, getattr(e, "status_code", "n/a"))
        raise LLMBackendError(type(e).__name__) from None
    return {"draft": text}


FIRST_TRY_COVERAGE = 0.8
RETRY_COVERAGE = 0.5


def keyword_coverage(question, context):
    """Fraction of the question's keywords that appear in the retrieved text."""
    keywords = tokenize(question)
    if not keywords:
        return 1.0
    context_words = set(tokenize(context))
    found = [w for w in keywords if w in context_words]
    return len(found) / len(keywords)


def review_answer(state: SupportState):
    draft = state.get("draft", "").strip()
    marker_used = INSUFFICIENT_MARKER in draft.upper().replace(" ", "_")
    required = FIRST_TRY_COVERAGE if state.get("retry_count", 0) == 0 else RETRY_COVERAGE
    low_coverage = keyword_coverage(state["question"], state.get("context", "")) < required
    if marker_used or low_coverage:
        return {"final_answer": ""}
    return {"final_answer": draft}


def route_after_review(state: SupportState):
    if state.get("final_answer"):
        return "finish"
    if state.get("retry_count", 0) < 1:
        return "retry"
    return "fallback"


def rewrite_query(state: SupportState):
    category_terms = {
        "returns_refunds": "return eligibility refund process",
        "order_tracking": "order dispatch tracking shipment status",
        "account_access": "account sign in password reset",
        "billing_payments": "billing payment transaction",
        "technical_support": "website technical troubleshooting",
        "general_support": "customer support help contact",
    }
    expanded = state["question"] + " " + category_terms.get(
        state.get("intent", "general_support"), "customer support")
    return {"search_query": expanded, "retry_count": state.get("retry_count", 0) + 1}


def fallback_answer(state: SupportState):
    return {"final_answer": FALLBACK_MESSAGE}


# ----------------------------------------------------------------------
# 6. BUILD THE GRAPH ONCE
# ----------------------------------------------------------------------
_builder = StateGraph(SupportState)
_builder.add_node("classify_intent", classify_intent)
_builder.add_node("retrieve", retrieve_context)
_builder.add_node("draft_answer", draft_answer)
_builder.add_node("review_answer", review_answer)
_builder.add_node("rewrite_query", rewrite_query)
_builder.add_node("fallback", fallback_answer)

_builder.add_edge(START, "classify_intent")
_builder.add_edge("classify_intent", "retrieve")
_builder.add_edge("retrieve", "draft_answer")
_builder.add_edge("draft_answer", "review_answer")
_builder.add_conditional_edges(
    "review_answer", route_after_review,
    {"finish": END, "retry": "rewrite_query", "fallback": "fallback"},
)
_builder.add_edge("rewrite_query", "retrieve")
_builder.add_edge("fallback", END)
support_graph = _builder.compile()


def ask_support(question: str) -> dict:
    """Run the graph for one question. Raises LLMBackendError if the provider fails."""
    return support_graph.invoke({"question": question, "retry_count": 0})
