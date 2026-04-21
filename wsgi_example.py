"""
wsgi_example.py
===============

Example WSGI entry point for deploying the ALMs demo on PythonAnywhere.

PythonAnywhere generates a WSGI file for every web app at
`/var/www/<YOUR_USERNAME>_pythonanywhere_com_wsgi.py` – open that file
from the **Web** tab (link labelled "WSGI configuration file") and replace
its contents with the block below.  Then hit **Reload** on the Web tab.

You do NOT need to copy this file to your app directory – it is a template.
"""

import os
import sys

# ── 1. Point Python at the app directory ─────────────────────────────────────
# Change `yourusername` and `alms-demo` to match your own setup.
APP_DIR = "/home/yourusername/alms-demo"
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

# ── 2. (Optional) server-side fallback credentials ───────────────────────────
# These are only used when the user leaves the UI fields blank.
# os.environ["OPENAI_API_KEY"]  = "sk-..."
# os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"

# ── 3. Import the WSGI callable (provided by app.py) ─────────────────────────
from app import application  # noqa: E402,F401
