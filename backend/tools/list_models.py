import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

if not config.GEMINI_API_KEY:
    raise SystemExit("GEMINI_API_KEY is not set in backend/.env")

from google import genai

client = genai.Client(api_key=config.GEMINI_API_KEY)
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions or not actions:
        print(f"{m.name:55} in={getattr(m, 'input_token_limit', '?')}")