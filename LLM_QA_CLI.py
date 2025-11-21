import os
import re
import argparse
import json
import requests

# Optional local settings loader: create `local_settings.py` (not checked into git)
# with variables like GOOGLE_API_KEY = "your_key_here" to avoid exporting env vars.
 

def preprocess(text: str) -> str:
    text = text.lower()
    # remove punctuation (basic)
    text = re.sub(r"[^\w\s]", "", text)
    tokens = text.split()
    return " ".join(tokens)

def construct_prompt(processed_question: str) -> str:
    return (
        "You are a helpful assistant. Answer concisely and clearly.\n"
        "Question: " + processed_question + "\nAnswer:"
    )

def send_to_gemini(prompt: str) -> str:
    """
    Send the prompt to Google's Generative Language API (Gemini-compatible).
    Requires `GOOGLE_API_KEY` env var. Optionally set `GEMINI_MODEL` (default: text-bison-001).
    """
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY (or GEMINI_API_KEY) not set")
    model = os.getenv("GEMINI_MODEL", "models/text-bison-001")
    # Build endpoint: https://generativelanguage.googleapis.com/v1/models/{model}:generateText?key=API_KEY
    base = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1/")
    # ensure trailing slash
    if not base.endswith("/"):
        base = base + "/"
    url = f"{base}{model}:generateText?key={api_key}"
    payload = {
        "prompt": {"text": prompt},
        "temperature": 0.2,
        "maxOutputTokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"Gemini API error: {resp.status_code} {resp.text}")
    data = resp.json()
    # Response usually has `candidates` with `output` or `candidates[0].output`.
    if isinstance(data, dict):
        # v1 and v1beta responses differ; try several common keys
        if "candidates" in data and isinstance(data["candidates"], list) and data["candidates"]:
            first = data["candidates"][0]
            if isinstance(first, dict) and "output" in first:
                return first["output"].strip()
        if "outputs" in data and isinstance(data["outputs"], list) and data["outputs"]:
            out0 = data["outputs"][0]
            if isinstance(out0, dict) and "content" in out0:
                # content may be list or string
                content = out0.get("content")
                if isinstance(content, list):
                    # join text pieces
                    parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
                    return "".join(parts).strip()
                return str(content).strip()
        # fallback: try top-level `output` or `text`
        if "output" in data:
            return str(data["output"]).strip()
        if "text" in data:
            return str(data["text"]).strip()
    # As a last resort, return the full JSON
    return json.dumps(data)

def send_to_huggingface(prompt: str) -> str:
    token = os.getenv("HUGGINGFACE_API_TOKEN")
    if not token:
        raise RuntimeError("HUGGINGFACE_API_TOKEN not set")
    url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
    headers = {"Authorization": f"Bearer {token}"}
    data = {"inputs": prompt}
    resp = requests.post(url, headers=headers, json=data, timeout=60)
    if not resp.ok:
        raise RuntimeError(f"HuggingFace API error: {resp.status_code} {resp.text}")
    out = resp.json()
    if isinstance(out, list) and out and isinstance(out[0], dict):
        return out[0].get("generated_text", json.dumps(out))
    return json.dumps(out)

def ask_llm(question: str) -> dict:
    processed = preprocess(question)
    prompt = construct_prompt(processed)
    errors = []
    try:
        answer = send_to_gemini(prompt)
        source = "gemini"
    except Exception as e:
        errors.append(str(e))
        try:
            answer = send_to_huggingface(prompt)
            source = "huggingface"
        except Exception as e2:
            errors.append(str(e2))
            answer = (
                "[SIMULATED RESPONSE — no API key configured] "
                "Processed question: " + processed
            )
            source = "simulated"
    return {"question": question, "processed": processed, "prompt": prompt, "answer": answer, "source": source, "errors": errors}

def main():
    parser = argparse.ArgumentParser(description="LLM Q&A CLI")
    parser.add_argument("-q", "--question", help="Question to ask the LLM", nargs="*")
    parser.add_argument("--api-key", help="Temporarily provide an API key for this run")
    parser.add_argument("--provider", choices=["gemini", "huggingface", "openai"], default="gemini", help="Which provider the --api-key belongs to")
    parser.add_argument("--model", help="Optional model name to use for Gemini (overrides GEMINI_MODEL env var)")
    args = parser.parse_args()
    if args.api_key:
        # set the key in-process so the send functions can pick it up
        if args.provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = args.api_key
            os.environ["GEMINI_API_KEY"] = args.api_key
        elif args.provider == "huggingface":
            os.environ["HUGGINGFACE_API_TOKEN"] = args.api_key
        elif args.provider == "openai":
            os.environ["OPENAI_API_KEY"] = args.api_key
    if args.model:
        os.environ["GEMINI_MODEL"] = args.model

    if args.question:
        question = " ".join(args.question)
    else:
        question = input("Enter your question: ")
    result = ask_llm(question)
    print("\n--- LLM Q&A Result ---")
    print("Original Question:", result["question"])
    print("Processed Question:", result["processed"])
    print("Source:", result["source"])
    print("\nAnswer:\n", result["answer"])

if __name__ == "__main__":
    main()