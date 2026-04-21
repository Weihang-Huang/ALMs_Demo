"""
ALMs – Authorial Language Models Demo (Huang, Murakami & Grieve, 2024/2025)
============================================================================

Attribution formula (from the paper):

    PPL(M, Q) = exp{ -(1/T) * Σ_{i=1..T} log P_M(x_i | x_{<i}) }
    predicted_author(Q) = argmin_{a ∈ candidates} PPL(M_a, Q)

where M_a is the *authorial* language model for candidate `a`.

This demo also implements the per-token **Comparative Negative Log-Likelihood**
(CNLL) visualisations from §"Token-level textual annotation" of the PLOS ONE
paper (Eq. 4–5, Figs 2–4, 7):

    • Extended CNLL:  CNLL_i(a) = NLL_i(a) − (1/(n−1)) Σ_{b≠a} NLL_i(b)
    • Blue cells  (CNLL < 0) = token i is EVIDENCE FOR author a.
    • Red  cells  (CNLL > 0) = token i is EVIDENCE AGAINST author a.
    • Colour saturation ∝ |CNLL|.

This demo talks to any OpenAI-compatible API (OpenAI, Ollama, vLLM, LocalAI,
Together, OpenRouter, …).  Each "authorial model" slot is (name, model,
primer) – the primer is a short public-domain excerpt that conditions the
base model on the author's style in lieu of full fine-tuning.

Two paths are used to obtain per-token log-probabilities, tried in order:

  1. /v1/completions with echo=True, max_tokens=0, logprobs=1
     – EXACT PPL per the paper formula.  Native per-token NLLs.
     – Works with local/open-weights models served by Ollama, vLLM, LocalAI,
       llama.cpp-server, etc.  NOT currently accepted by OpenAI's hosted
       models (they forbid echo+logprobs together).

  2. /v1/chat/completions with logprobs=True, top_logprobs=20
     – APPROXIMATE PPL via "predict-next-word" scoring across the
       questioned text, one API call per (author, word).
     – For each word position i, the model is asked (conditioned on the
       author's primer and words[:i]) for its next-token distribution; we
       read off the log-probability of the actual word from `top_logprobs`
       and record it as NLL_i(a).
     – Works with gpt-4o-mini, gpt-4.1-nano, and most chat models that
       allow logprobs.  (gpt-5-nano currently blocks logprobs → 403.)

Files (FLAT – no subfolders):
    app.py  index.html  style.css  script.js
    requirements.txt  README.md  .env.example  wsgi_example.py

PythonAnywhere deployment: `application = app` is exposed at module level
so the WSGI file can simply `from app import application`.
"""

from __future__ import annotations

import math
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

# Optional: load .env in local dev.  Harmless on PythonAnywhere.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Flask app setup
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=BASE_DIR,
    static_folder=None,  # we serve static files via explicit routes
)

# PythonAnywhere's WSGI file does `from app import application`.
application = app


# ──────────────────────────────────────────────────────────────────────────────
# Built-in author primers & demo texts (all public domain)
# ──────────────────────────────────────────────────────────────────────────────

# Default to a chat model that reliably exposes logprobs on OpenAI.
# Users can swap this in the UI (e.g. gpt-4.1-nano, or a local model name).
DEFAULT_MODEL = "gpt-4o-mini"

DEFAULT_AUTHORS: list[dict[str, str]] = [
    {
        "name": "Jane Austen",
        "model": DEFAULT_MODEL,
        "primer": (
            "It is a truth universally acknowledged, that a single man in "
            "possession of a good fortune, must be in want of a wife. "
            "However little known the feelings or views of such a man may "
            "be on his first entering a neighbourhood, this truth is so "
            "well fixed in the minds of the surrounding families, that he "
            "is considered the rightful property of some one or other of "
            "their daughters. Miss Elizabeth Bennet, handsome, clever, and "
            "lively, had a lively disposition, which delighted in any "
            "thing ridiculous. Catherine Morland, in training to be a "
            "heroine, was pleasant-looking, with a good-humoured face. "
            "Anne Elliot, with an elegance of mind and sweetness of "
            "character, was nobody with either father or sister; her word "
            "had no weight; her convenience was always to give way; she "
            "was only Anne. Miss Dashwood possessed a strength of "
            "understanding, and coolness of judgment, which qualified her, "
            "though only nineteen, to be the counsellor of her mother. "
            "Elinor, in giving her consent, felt a very considerable "
            "share of satisfaction."
        ),
    },
    {
        "name": "Arthur Conan Doyle",
        "model": DEFAULT_MODEL,
        "primer": (
            "'You have been in Afghanistan, I perceive,' said Sherlock "
            "Holmes. 'How on earth did you know that?' I asked in "
            "astonishment. It is a capital mistake to theorise before one "
            "has data. Insensibly one begins to twist facts to suit "
            "theories, instead of theories to suit facts. You see, "
            "Watson, but you do not observe. 'The game is afoot,' cried "
            "Holmes, as he sprang from his chair. Come, Watson, come! "
            "Quick, man, if you love me. The hound, Watson! The gigantic "
            "hound of the Baskervilles! Mr. Jabez Wilson here has been "
            "good enough to call upon me this morning, and to begin a "
            "narrative which promises to be one of the most singular."
        ),
    },
    {
        "name": "Mark Twain",
        "model": DEFAULT_MODEL,
        "primer": (
            "You don't know about me without you have read a book by the "
            "name of The Adventures of Tom Sawyer; but that ain't no "
            "matter. That book was made by Mr. Mark Twain, and he told the "
            "truth, mainly. Tom appeared on the sidewalk with a bucket of "
            "whitewash and a long-handled brush. He surveyed the fence, "
            "and all gladness left him. Saturday morning was come, and all "
            "the summer world was bright and fresh, and brimming with "
            "life. Tom's mouth watered for the apple, but he stuck to his "
            "work. The boys of the village came by in groups. Presently, "
            "the one boy amongst them all that Tom dreaded most came "
            "along. Huckleberry was cordially hated and dreaded by all "
            "the mothers of the town, because he was idle, and lawless, "
            "and vulgar and bad."
        ),
    },
    {
        "name": "William Shakespeare",
        "model": DEFAULT_MODEL,
        "primer": (
            "All the world's a stage, and all the men and women merely "
            "players; they have their exits and their entrances, and one "
            "man in his time plays many parts. To be, or not to be, that "
            "is the question: whether 'tis nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune, or to take arms "
            "against a sea of troubles. Friends, Romans, countrymen, lend "
            "me your ears; I come to bury Caesar, not to praise him. Now "
            "is the winter of our discontent made glorious summer by this "
            "sun of York. O, wherefore art thou Romeo? Deny thy father and "
            "refuse thy name. Hark, hark! the lark at heaven's gate sings."
        ),
    },
]

DEMO_TEXTS: dict[str, str] = {
    "austen": (
        "The family of Dashwood had been long settled in Sussex. Their "
        "estate was large, and their residence was at Norland Park, in "
        "the centre of their property, where, for many generations, they "
        "had lived in so respectable a manner as to engage the general "
        "good opinion of their surrounding acquaintance."
    ),
    "doyle": (
        "Sherlock Holmes took his bottle from the corner of the mantel-piece "
        "and his hypodermic syringe from its neat morocco case. With his "
        "long, white, nervous fingers he adjusted the delicate needle, and "
        "rolled back his left shirt-cuff."
    ),
    "twain": (
        "Aunt Polly she told me all about the good place, and I said I "
        "wished I was there. She said it was wicked to say what I said; "
        "said she wouldn't say it for the whole world; she was going to "
        "live so as to go to the good place. Well, I couldn't see no "
        "advantage in going where she was going, so I made up my mind I "
        "wouldn't try for it."
    ),
    "shakespeare": (
        "What a piece of work is a man! how noble in reason! how infinite in "
        "faculty! in form and moving how express and admirable! in action how "
        "like an angel! in apprehension how like a god! the beauty of the "
        "world, the paragon of animals!"
    ),
}

# Number of word positions sampled for the chat-based PPL (fast mode).
# Full mode scores every word and is needed for token-level visualisations.
CHAT_SCORING_K = 6

# Parallel API calls when scoring every word.  ~8 keeps latency bearable
# without hammering the rate limit or triggering 5xx errors on low-tier keys.
MAX_WORKERS = 8

# Safety cap on the number of words that will be scored per author in full
# (token-level) mode.  Beyond this, we fall back to K-sampled mode.
MAX_TOKEN_VIZ_WORDS = 120


# ──────────────────────────────────────────────────────────────────────────────
# OpenAI client factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_client(api_key: str, base_url: str | None):
    if openai is None:
        raise RuntimeError(
            "The `openai` package is not installed. "
            "Run `pip install -r requirements.txt`."
        )
    kwargs: dict[str, Any] = {"api_key": api_key, "timeout": 60.0}
    if base_url and base_url.strip():
        kwargs["base_url"] = base_url.strip().rstrip("/")
    return openai.OpenAI(**kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Word tokenisation (shared by both paths and by the viz layer)
# ──────────────────────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r"\S+")


def _tokenise_with_spans(text: str) -> list[dict]:
    """Split text into whitespace-separated 'words' with char spans."""
    return [
        {"word": m.group(), "start": m.start(), "end": m.end()}
        for m in _WORD_RE.finditer(text)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Path 1 – EXACT PPL via /v1/completions + echo
# ──────────────────────────────────────────────────────────────────────────────

def _ppl_via_completions_echo(text: str, primer: str, model: str, client) -> dict:
    """
    Exact ALMs PPL via the legacy /v1/completions endpoint with echo=True.

    Works with local/open-weights models (Ollama, vLLM, LocalAI, llama.cpp,
    TGI, …).  OpenAI-hosted models currently refuse echo+logprobs together.

    The primer is prepended for style conditioning; only questioned-text
    tokens contribute to the reported PPL.  We also return per-token NLLs
    (aligned to whitespace words) for the CNLL visualisations.
    """
    primer = (primer or "").strip()
    text = text.strip()

    if primer:
        sep = "\n\n"
        full_prompt = primer + sep + text
        text_char_start = len(primer) + len(sep)
    else:
        full_prompt = text
        text_char_start = 0

    resp = client.completions.create(
        model=model,
        prompt=full_prompt,
        max_tokens=0,
        echo=True,
        logprobs=1,
    )

    lp = resp.choices[0].logprobs
    token_logprobs = lp.token_logprobs or []   # list[float | None]
    sub_tokens     = lp.tokens or []
    text_offsets   = lp.text_offset or []

    if text_char_start > 0 and text_offsets:
        first_text_tok = next(
            (i for i, off in enumerate(text_offsets) if off >= text_char_start),
            len(text_offsets),
        )
    else:
        first_text_tok = 0

    # Aggregate sub-token NLLs onto whitespace words for the viz layer.
    words = _tokenise_with_spans(text)
    word_nlls: list[float] = [0.0] * len(words)
    word_valid: list[bool] = [False] * len(words)

    for i in range(first_text_tok, len(token_logprobs)):
        lp_val = token_logprobs[i]
        if lp_val is None:
            continue
        tok_offset = text_offsets[i] - text_char_start if i < len(text_offsets) else -1
        if tok_offset < 0:
            continue
        # Find which word this sub-token falls into.
        for wi, w in enumerate(words):
            if w["start"] <= tok_offset < w["end"]:
                word_nlls[wi] += -lp_val
                word_valid[wi] = True
                break

    valid_nlls = [nll for nll, ok in zip(word_nlls, word_valid) if ok]
    if not valid_nlls:
        raise ValueError("API returned no usable token log-probabilities.")

    # Per-token record for the viz layer.
    per_token = [
        {"word": w["word"], "nll": (word_nlls[wi] if word_valid[wi] else None)}
        for wi, w in enumerate(words)
    ]

    mean_nll = sum(valid_nlls) / len(valid_nlls)
    return {
        "ppl":       round(math.exp(mean_nll), 4),
        "n_tokens":  len(valid_nlls),
        "method":    "completions/echo",
        "per_token": per_token,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Path 2 – APPROXIMATE PPL via chat top_logprobs word-scoring
# ──────────────────────────────────────────────────────────────────────────────

def _sample_positions(n_words: int, k: int) -> list[int]:
    """K evenly-spaced word positions in [1, n_words-1]."""
    if n_words < 2:
        return []
    step = max(1, n_words // (k + 1))
    return [i * step for i in range(1, k + 1) if 0 < i * step < n_words]


def _score_one_position(words: list[str], i: int, system_msg: str,
                        model: str, client) -> tuple[int, float | None, bool]:
    """
    Score a single word position.  Returns (i, nll, was_hit).

    Raises openai.AuthenticationError / openai.RateLimitError so the caller
    can fail fast; silences other exceptions (returns (i, None, False)).
    Retries once on transient 5xx server errors.
    """
    # Use a TRAILING-SPACE prefix so the model predicts the next whole token
    # rather than just the space character.  The OpenAI chat API tokenises
    # its input and the 'next token' would otherwise almost always be " ".
    prefix = " ".join(words[:i]) + " "
    actual_raw = words[i]
    actual = actual_raw.strip(".,;:!?\"'()[]—–-").lower()
    if not actual:
        return (i, None, False)

    resp = None
    for attempt in range(2):          # one retry on transient errors
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user",   "content": prefix},
                ],
                max_tokens=1,
                logprobs=True,
                top_logprobs=20,
                temperature=0,
                seed=0,                # deterministic across runs
            )
            break
        except openai.AuthenticationError:
            raise
        except openai.RateLimitError:
            raise
        except (openai.APIError, openai.APIConnectionError):
            continue
        except Exception:
            break
    if resp is None:
        return (i, None, False)

    content = resp.choices[0].logprobs and resp.choices[0].logprobs.content
    if not content:
        return (i, None, False)
    tops = content[0].top_logprobs

    # Match the actual word (case-insensitive) against the top-20 tokens.
    # Accept both exact-prefix and starts-with matches to tolerate the
    # tokenizer splitting words into BPE pieces.
    hit = None
    for tlp in tops:
        tok = tlp.token.strip().lower()
        if not tok:
            continue
        if actual.startswith(tok) or tok.startswith(actual):
            hit = tlp.logprob
            break

    if hit is None:
        # Penalty: two nats below the worst observed top-20 logprob.
        worst = min(tlp.logprob for tlp in tops) if tops else -5.0
        return (i, -(worst - 2.0), False)
    return (i, -hit, True)


def _ppl_via_chat_scoring(
    text: str,
    primer: str,
    model: str,
    client,
    full_token_viz: bool = False,
    k: int = CHAT_SCORING_K,
) -> dict:
    """
    Approximate PPL via "predict-next-word" scoring.

    Two modes:

      • full_token_viz=True – every word position is scored, in parallel
        via a thread pool.  Returns per-token NLLs for the CNLL viz layer.

      • full_token_viz=False – only K evenly-spaced positions are scored.
        Faster, but the per_token array is sparse.
    """
    words_raw = _WORD_RE.findall(text.strip())
    n = len(words_raw)
    if n < 3:
        raise ValueError("Text too short for chat-based scoring (need ≥3 words).")

    if full_token_viz and n <= MAX_TOKEN_VIZ_WORDS:
        positions = list(range(1, n))   # score every word after the first
    else:
        positions = _sample_positions(n, k)
        if not positions:
            raise ValueError("Could not sample any word positions.")

    primer = (primer or "").strip()
    system_msg = (
        f"Style reference:\n{primer}\n\nContinue the text in the same style."
        if primer else
        "Continue the text naturally."
    )

    # Parallel per-position scoring.
    results: dict[int, float | None] = {}
    n_misses = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {
            pool.submit(_score_one_position, words_raw, i, system_msg, model, client): i
            for i in positions
        }
        for fut in as_completed(futs):
            idx, nll, hit = fut.result()
            results[idx] = nll
            if nll is not None and not hit:
                n_misses += 1

    valid_nlls = [v for v in results.values() if v is not None]
    if not valid_nlls:
        raise ValueError("Could not score any word positions (all API calls failed).")

    # Per-token array – word 0 has no conditioning, so nll=None there.
    per_token = []
    for i, w in enumerate(words_raw):
        per_token.append({
            "word": w,
            "nll": results.get(i),     # None for word 0 and unscored positions
        })

    mean_nll = sum(valid_nlls) / len(valid_nlls)
    return {
        "ppl":        round(math.exp(mean_nll), 4),
        "n_tokens":   len(valid_nlls),
        "method":     "chat/word-scoring≈",
        "n_misses":   n_misses,
        "per_token":  per_token,
        "note": (
            f"Approx ALMs PPL – mean NLL of {len(valid_nlls)} scored words "
            f"(of which {n_misses} fell outside the model's top-20). "
            "For the exact paper formula use a model that supports "
            "/v1/completions with echo=True (Ollama, vLLM, LocalAI, …)."
        ),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Top-level dispatcher
# ──────────────────────────────────────────────────────────────────────────────

def compute_perplexity(
    text: str,
    author: dict,
    api_key: str,
    base_url: str,
    full_token_viz: bool = False,
) -> dict:
    """Compute PPL for one authorial slot, trying echo then chat-scoring."""
    if openai is None:
        raise RuntimeError("openai package missing – install requirements.txt")

    model  = (author.get("model")  or DEFAULT_MODEL).strip()
    primer = (author.get("primer") or "").strip()
    name   = author.get("name")    or "Unknown"

    client = _make_client(api_key, base_url)

    primary_err: Exception | None = None

    # Path 1: completions + echo (exact) — always full per-token.
    try:
        r = _ppl_via_completions_echo(text, primer, model, client)
        r["author"] = name
        r["model"]  = model
        return r
    except openai.AuthenticationError:
        raise
    except openai.RateLimitError:
        raise
    except Exception as exc:
        primary_err = exc

    # Path 2: chat top_logprobs scoring (approx)
    try:
        r = _ppl_via_chat_scoring(
            text, primer, model, client, full_token_viz=full_token_viz,
        )
        r["author"] = name
        r["model"]  = model
        r["fallback_reason"] = str(primary_err)[:220] if primary_err else ""
        return r
    except openai.AuthenticationError:
        raise
    except openai.RateLimitError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"Both perplexity methods failed for model '{model}'. "
            f"completions/echo: {primary_err}. chat/word-scoring: {exc}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# CNLL – Comparative Negative Log-Likelihood (Eq. 5 of the PLOS ONE paper)
# ──────────────────────────────────────────────────────────────────────────────

def _build_cnll_matrix(results: list[dict]) -> list[dict]:
    """
    For each word position i and each candidate author a, compute

        CNLL_i(a) = NLL_i(a) − (1/(n−1)) · Σ_{b≠a} NLL_i(b)

    Returns a list of {word, nlls: {author: nll or None},
                       cnll: {author: cnll or None}} aligned to the word list.
    """
    if not results:
        return []

    # Words come from the first author's per_token array (all authors share
    # the same tokenisation since they score the same questioned text).
    ref = next((r for r in results if r.get("per_token")), None)
    if ref is None:
        return []
    words = [t["word"] for t in ref["per_token"]]
    n_words = len(words)
    author_names = [r["author"] for r in results]

    out: list[dict] = []
    for i in range(n_words):
        nlls: dict[str, float | None] = {}
        for r in results:
            pt = r.get("per_token") or []
            nlls[r["author"]] = pt[i]["nll"] if i < len(pt) else None

        valid = {a: v for a, v in nlls.items() if v is not None}
        cnll: dict[str, float | None] = {a: None for a in author_names}
        if len(valid) >= 2:
            for a, v in valid.items():
                others = [v2 for a2, v2 in valid.items() if a2 != a]
                if others:
                    cnll[a] = v - (sum(others) / len(others))

        out.append({"word": words[i], "nlls": nlls, "cnll": cnll})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def _css():
    return send_from_directory(BASE_DIR, "style.css", mimetype="text/css")


@app.route("/script.js")
def _js():
    return send_from_directory(BASE_DIR, "script.js",
                               mimetype="application/javascript")


@app.route("/default_authors")
def _default_authors():
    return jsonify(DEFAULT_AUTHORS)


@app.route("/demo_texts")
def _demo_texts():
    return jsonify(DEMO_TEXTS)


@app.route("/healthz")
def _healthz():
    return jsonify({"ok": True, "openai_installed": openai is not None})


@app.route("/attribute", methods=["POST"])
def attribute():
    """
    POST /attribute
    Request body (JSON):
        {
          "text":             "<questioned text>",
          "api_key":          "<OpenAI-compatible key>",   (optional – env fallback)
          "base_url":         "<endpoint>",                (optional – env fallback)
          "authors":          [{"name": "...", "model": "...", "primer": "..."}, ...],
          "token_viz":        true    // default true; set false for fast mode
        }

    API key and base URL are used *per request only* – never logged or
    persisted on the server.

    Response adds `tokens`: a list aligned to whitespace-separated words in
    the questioned text, carrying per-author NLL and per-author CNLL values,
    for the color-coded text annotation (Fig 2/4) and heatmap (Fig 3/7).
    """
    if openai is None:
        return jsonify({"error": "openai package is not installed on server."}), 500

    data = request.get_json(silent=True) or {}

    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please provide a questioned text."}), 400
    if len(text) > 10_000:
        return jsonify({"error": "Text exceeds 10,000-character limit for this demo."}), 400

    api_key = ((data.get("api_key") or "").strip()
               or os.environ.get("OPENAI_API_KEY", ""))
    base_url = ((data.get("base_url") or "").strip()
                or os.environ.get("OPENAI_BASE_URL", ""))

    if not api_key:
        return jsonify({
            "error": ("No API key provided. Paste your key in the "
                      "'Your OpenAI-compatible API key' field above.")
        }), 401

    authors = data.get("authors") or DEFAULT_AUTHORS
    if not isinstance(authors, list) or not authors:
        return jsonify({"error": "No author slots configured."}), 400

    token_viz = bool(data.get("token_viz", True))

    results: list[dict] = []
    errors:  list[dict] = []

    for author in authors:
        name  = (author.get("name")  or "").strip()
        model = (author.get("model") or "").strip()
        if not name or not model:
            continue
        try:
            results.append(compute_perplexity(
                text, author, api_key, base_url,
                full_token_viz=token_viz,
            ))
        except openai.AuthenticationError:
            return jsonify({
                "error": "Invalid or expired API key. Check your key and retry."
            }), 401
        except openai.RateLimitError:
            return jsonify({
                "error": ("Rate limit reached at the upstream provider. "
                          "Wait a moment and try again.")
            }), 429
        except openai.NotFoundError:
            errors.append({
                "author": name,
                "error": f"Model '{model}' not found for this endpoint/key.",
            })
        except openai.APIConnectionError as exc:
            errors.append({
                "author": name,
                "error": f"Network error reaching the API: {exc}",
            })
        except openai.PermissionDeniedError as exc:
            # Typical for models that block logprobs (e.g. gpt-5-nano).
            errors.append({
                "author": name,
                "error": (f"Model '{model}' refused the request "
                          f"(often logprobs are disabled): {exc}"),
            })
        except Exception as exc:
            errors.append({"author": name, "error": str(exc)[:400]})

    if not results:
        return jsonify({
            "error": "All author models failed to return a perplexity score.",
            "details": errors,
        }), 502

    # Lowest PPL = predicted author.
    results.sort(key=lambda r: r["ppl"])
    winner = results[0]["author"]

    # Build the CNLL matrix for viz (Eq. 5).  Strip heavy `per_token` arrays
    # from the per-author payload – the same info is now in `tokens`.
    tokens = _build_cnll_matrix(results)
    for r in results:
        r.pop("per_token", None)

    return jsonify({
        "results": results,
        "winner":  winner,
        "tokens":  tokens,
        "errors":  errors,
    })


# ──────────────────────────────────────────────────────────────────────────────
# Dev entry point.  PythonAnywhere never executes this block – it imports
# `application` directly via the WSGI file instead.
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
