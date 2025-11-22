from flask import Flask, render_template, request
import logic
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""
    results = None

    if request.method == "POST":
        original_text = request.form.get("user_input", "").strip()

        if original_text:
            lines = [line.strip() for line in original_text.splitlines() if line.strip()]
            # استدعاء الفنكشن من داخل logic
            results = logic.run_model_on_input(lines)

    return render_template(
        "index.html",
        original_text=original_text,
        results=results
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


