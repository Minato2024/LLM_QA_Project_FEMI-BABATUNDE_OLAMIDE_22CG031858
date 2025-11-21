from flask import Flask, render_template, request
import os
import traceback

# Import ask_llm from CLI module
from LLM_QA_CLI import ask_llm

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    result = None
    error = None
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            try:
                result = ask_llm(question)
            except Exception as e:
                error = str(e) + "\n" + traceback.format_exc()
    return render_template("index.html", question=question, result=result, error=error)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)