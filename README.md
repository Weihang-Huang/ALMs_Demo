# ALMs Demo — Authorial Language Models for Authorship Attribution

A minimal, single-page Flask demo that illustrates the
**Authorial Language Models (ALMs)** method from
[Huang, Murakami & Grieve (PLOS ONE 2025 / arXiv 2401.12005)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327081):

> Given candidate authors *a ∈ C*, each represented by a causal
> language model *M<sub>a</sub>* further pre-trained on that author's
> writings, the questioned document *Q* is attributed to
> `î = argmin_{a ∈ C} PPL(M_a, Q)`, where
>
> `PPL(M, Q) = exp{ −(1/T) Σ_{i=1..T} log P_M(x_i | x_{<i}) }`.

The demo targets an **OpenAI-compatible API** (OpenAI, Ollama, vLLM,
LocalAI, OpenRouter, Together, …). Each "authorial model" slot is a
triple **(author name, model name, style primer)**; the primer is a
short public-domain excerpt that conditions the base model on the
author's style in lieu of full fine-tuning.

It is deliberately minimal — **this is a teaching demo, not production
code.**

---

## Repository layout (flat – no subfolders)

```
app.py              Flask backend (exposes `application = app` for WSGI)
index.html          Single-page UI
style.css           Styles
script.js           Vanilla-JS frontend logic
requirements.txt    Python deps (Flask, openai, python-dotenv)
README.md           You are here
.env.example        Optional server-side credential fallback
wsgi_example.py     Reference WSGI file for PythonAnywhere
```

---

## Quick start (local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Optional server-side fallback:
cp .env.example .env && $EDITOR .env

python app.py
# → http://127.0.0.1:5000
```

Open the page, paste your API key in the UI, pick a demo text, and hit
**Attribute authorship**.

---

## Deploy to PythonAnywhere — step-by-step

PythonAnywhere is a Python-first shared-hosting platform. This app runs
on both the **Free tier** and any paid tier.

1. **Create / log into** an account at
   [pythonanywhere.com](https://www.pythonanywhere.com/).

2. **Upload the files.** From the *Files* tab, create a directory (e.g.
   `alms-demo`) and upload every file in this repo into it. The flat
   file structure means no extra folders are required.

   Or, from a Bash console:
   ```bash
   cd ~
   git clone <this-repo>.git alms-demo
   # or: unzip alms-demo.zip -d ~/alms-demo
   ```

3. **Create a virtualenv and install dependencies** in a Bash console
   (on PythonAnywhere, Python 3.10 is the current minimum supported
   version):
   ```bash
   cd ~/alms-demo
   python3.10 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Create a web app.** Go to the **Web** tab → *Add a new web app* →
   *Manual configuration* → *Python 3.10*.

5. **Point the web app at your code.** Still on the Web tab:
   - *Source code*: `/home/<YOUR_USERNAME>/alms-demo`
   - *Working directory*: `/home/<YOUR_USERNAME>/alms-demo`
   - *Virtualenv*: `/home/<YOUR_USERNAME>/alms-demo/.venv`

6. **Edit the WSGI file.** Click the *WSGI configuration file* link on
   the Web tab. Delete its contents and paste (adjusting the username):

   ```python
   import sys
   APP_DIR = "/home/<YOUR_USERNAME>/alms-demo"
   if APP_DIR not in sys.path:
       sys.path.insert(0, APP_DIR)
   from app import application
   ```

   `wsgi_example.py` in this repo shows the full template with optional
   server-side credential variables.

7. **(Optional) set server-side env vars.** If you want a fallback key
   for users who leave the UI field empty, add lines like
   `os.environ["OPENAI_API_KEY"] = "sk-..."` to the WSGI file
   before the `from app import application` line. Do **not** commit real
   keys to a public repo.

8. **Reload** the web app (big green button on the Web tab). Visit your
   URL – e.g. `https://<YOUR_USERNAME>.pythonanywhere.com` – and use the
   demo.

### ⚠️ Free-tier outbound-network whitelist

PythonAnywhere **free accounts can only make outbound HTTP(S) requests
to hosts on a whitelist**. `api.openai.com` is on the whitelist, so the
demo works out of the box with stock OpenAI keys. Providers **not** on
the whitelist — OpenRouter, Together, Groq, local tunnels, most
Ollama/vLLM instances — require a **paid tier** (Hacker plan or above),
which removes the restriction. See the
[PythonAnywhere whitelist](https://www.pythonanywhere.com/whitelist/)
for the current list, and request additions via their forums if needed.

### Static files (optional optimisation)

Flask serves `style.css` and `script.js` directly, which is fine for a
demo. For slightly better performance you can add a static-file mapping
on the Web tab:

| URL   | Directory                                |
|-------|------------------------------------------|
| `/style.css` | `/home/<USER>/alms-demo/style.css`       |
| `/script.js` | `/home/<USER>/alms-demo/script.js`       |

(PythonAnywhere will serve these without hitting Python at all.)

---

## Using your own API key

The UI asks for **two** credentials per request:

1. **Your OpenAI-compatible API key** — any `sk-...` style key from
   OpenAI, OpenRouter, Together, Groq, Fireworks, …  or an arbitrary
   non-empty string for local Ollama / vLLM / LocalAI servers that do
   not check auth.
2. **Base URL** *(optional)* — the provider endpoint. Defaults to
   `https://api.openai.com/v1` when blank. Examples:
   - `https://api.openai.com/v1`
   - `https://openrouter.ai/api/v1`
   - `https://api.together.xyz/v1`
   - `http://localhost:11434/v1`  *(Ollama)*
   - `http://localhost:8000/v1`   *(vLLM / LocalAI)*

**Privacy note.** The key is kept in the current browser session and
forwarded only to the model endpoint you configure. `app.py` never
logs it, never persists it, and never echoes it back in a response.

**Env-var fallback.** If a user leaves the UI fields blank, the server
falls back to `OPENAI_API_KEY` / `OPENAI_BASE_URL` env vars (if set).
Leave them unset to force every user to supply their own key.

### Which model to choose?

The app implements **two** perplexity paths, tried in order:

1. **Exact path — `/v1/completions` with `echo=True, max_tokens=0,
   logprobs=1`.** Yields exact per-token logprobs of the questioned
   text. Works with any local/open-weights model served by Ollama,
   vLLM, LocalAI, llama.cpp-server, TGI, etc.  *OpenAI's hosted models
   currently refuse `echo` + `logprobs` together*, so this path only
   fires against local/compatible servers.

2. **Approximate path — `/v1/chat/completions` with `logprobs=True,
   top_logprobs=20`.** For K evenly-spaced word positions in the
   questioned text, the app asks the model (conditioned on the
   author's primer) for its next-token distribution and reads off the
   log-probability of the *actual* next word from `top_logprobs`.
   PPL ≈ exp(mean NLL over sampled positions).  Works with
   `gpt-4o-mini` (default), `gpt-4.1-nano`, `gpt-4o`, and most chat
   models that expose logprobs.

   Rows served this way are tagged `chat/word-scoring≈` in the
   results table and run in ~5–7 s per author per text.

---

## Token-level visualisations (CNLL)

The demo also reproduces the two token-level diagnostic views from
Figures 2–4 and 7 of the PLOS ONE paper:

1. **Colour-coded text annotation.** Every word is tinted by its
   **comparative NLL (CNLL)** under the selected author model:

   `CNLLᵢ(a) = NLLᵢ(a) − (1 / (|C|−1)) · Σ_{b≠a} NLLᵢ(b)`

   - **Blue** = the selected author predicted this word *better* than
     the average competitor (evidence **for** that author).
   - **Red** = *worse* than average (evidence **against**).
   - Saturation encodes magnitude, clipped to a visual `v_max`.

   A dropdown lets you flip between authors to compare. This matches
   the colour scheme used in the published figures.

2. **CNLL heat-map.** A tokens-×-authors grid where each cell is
   coloured by CNLL under that author's ALM — rows = words of the
   questioned text (in order), columns = candidate authors, winner
   first (outlined in red). Below the grid is a **mean-CNLL bar
   chart**: the author with the most negative mean wins, echoing the
   `argmin PPL` decision rule. This matches Figs. 3 & 7 in the paper.

### How the per-token NLL is obtained

- **Completions / echo path** (local models): directly extracts the
  per-sub-token logprobs returned by `/v1/completions` with
  `echo=True, logprobs=1`, then aggregates sub-tokens into whitespace
  words.
- **Chat / word-scoring path** (OpenAI hosted): for each word
  position, sends a separate `/v1/chat/completions` request with the
  prefix so far (trailing space, `max_tokens=1, logprobs=True,
  top_logprobs=20, temperature=0, seed=0`) and reads the logprob of
  the actual next word from `top_logprobs`. Positions are scored in
  parallel (8 workers) with a single retry on transient 5xx errors.
  To keep latency reasonable, the visualisation caps at the first
  `MAX_TOKEN_VIZ_WORDS = 120` words.

### Enabling the visualisations

Tick the **"Token-level visualisation (CNLL)"** checkbox before
clicking *Attribute authorship*. The attribution summary, annotated
text, author picker, and heat-map all render on the same page.
Leaving the box unchecked keeps the demo fast (document-level PPL
only).

**Note on `gpt-5-nano`:** at the time of writing, OpenAI returns
`403 – You are not allowed to request logprobs from this model` for
`gpt-5-nano`, so that model cannot be used here.  Pick `gpt-4o-mini`
or `gpt-4.1-nano` instead (both are inexpensive and expose logprobs).

---

## API — `POST /attribute`

```json
{
  "text":    "Emma Woodhouse, handsome, clever, and rich …",
  "api_key": "sk-...",
  "base_url": "https://api.openai.com/v1",
  "authors": [
    {"name": "Jane Austen", "model": "gpt-3.5-turbo-instruct",
     "primer": "It is a truth universally acknowledged …"},
    {"name": "Arthur Conan Doyle", "model": "gpt-3.5-turbo-instruct",
     "primer": "I had called upon my friend Mr. Sherlock Holmes …"}
  ]
}
```

Response:

```json
{
  "winner": "Jane Austen",
  "results": [
    {"author": "Jane Austen",        "model": "…", "ppl": 6.14, "n_tokens": 58, "method": "completions/echo"},
    {"author": "Arthur Conan Doyle", "model": "…", "ppl": 7.02, "n_tokens": 58, "method": "completions/echo"}
  ],
  "errors": []
}
```

Error codes:

| HTTP | Meaning                                   |
|------|-------------------------------------------|
| 400  | Missing/invalid text, empty author list   |
| 401  | Missing or invalid API key                |
| 429  | Provider rate-limit hit                   |
| 502  | All author models failed                  |

---

## Limitations

- **This demo does not fine-tune models.** Each author's "ALM" is
  approximated by prompt-conditioning a shared base model on a style
  primer. True ALMs (one fine-tune per author) will show much sharper
  perplexity differences. The algorithmic structure is faithful: pick
  the author whose conditioned LM produces the lowest PPL on Q.
- Questioned text capped at 10 000 characters.
- Synchronous request handling — each author = one blocking API call.
  Plenty fast enough for a handful of authors and a page of text; don't
  use it as-is for batch benchmarks.
- No background threads, no writes outside the app directory, no
  `app.run()` in the WSGI path — i.e. clean PythonAnywhere citizen.

---

## References

- Huang, W., Murakami, A. & Grieve, J. (2025). *Attributing authorship
  via the perplexity of authorial language models.* PLOS ONE.
  [doi:10.1371/journal.pone.0327081](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0327081)
- Huang, W. & Grieve, J. (2024). *ALMs: Authorial Language Models for
  Authorship Attribution.* [arXiv:2401.12005](https://arxiv.org/abs/2401.12005)
- Reference implementation (GPT-2 fine-tuning):
  [github.com/Weihang-Huang/ALMs](https://github.com/Weihang-Huang/ALMs)

---

## License

MIT — this is a teaching demo; use it freely.
