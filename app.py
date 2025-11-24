from flask import Flask, render_template, request, send_file
import logic
import io
import os
import tempfile
import pandas as pd
import time

app = Flask(__name__)

# نخزن آخر نتائج هنا لكي نقدر نعمل download
last_results = None


@app.route("/", methods=["GET", "POST"])
def index():
    global last_results

    results = None
    original_text = ""
    processing_time = None  # وقت الحساب

    if request.method == "POST":
        original_text = request.form.get("user_input", "")

        lines = []

        # -------- من المربع النصي --------
        if original_text:
            for line in original_text.splitlines():
                line = line.strip()
                if line:
                    lines.append(line)

        # -------- من ملف مرفوع --------
        uploaded = request.files.get("molecule_file")
        if uploaded and uploaded.filename:
            filename = uploaded.filename
            ext = filename.rsplit(".", 1)[-1].lower()

            try:
                # ملفات نصية / CSV
                if ext in ("txt", "smi", "smiles", "csv"):
                    text_data = uploaded.read().decode("utf-8", errors="ignore")
                    for line in text_data.splitlines():
                        if line.strip():
                            lines.append(line.strip())

                # ملفات Excel
                elif ext in ("xlsx", "xls"):
                    file_bytes = uploaded.read()
                    df = pd.read_excel(io.BytesIO(file_bytes))

                    # تحديد العمود الصحيح
                    col = None
                    for cand in ["SMILES", "smiles", "Smiles", "Mol", "Name"]:
                        if cand in df.columns:
                            col = cand
                            break
                    if col is None:
                        col = df.columns[0]

                    for val in df[col].astype(str):
                        if val.strip():
                            lines.append(val.strip())

                # ملفات SDF
                elif ext in ("sdf", "sdf.gz"):
                    suffix = "." + ext
                    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
                    os.close(fd)

                    try:
                        uploaded.save(tmp_path)
                        mols = logic.load_molecules(tmp_path, max_mols=logic.MAX_MOLS)
                        for smi, m in mols:
                            smi = str(smi).strip()
                            if smi:
                                lines.append(smi)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

            except Exception as e:
                results = [{
                    "input_text": "File error",
                    "drug_likeness": "Error",
                    "details": f"Could not read file: {e}",
                }]

        # -------- تشغيل الموديل --------
        if lines and results is None:
            start = time.time()
            results = logic.run_model_on_input(lines)
            end = time.time()

            processing_time = end - start  # الوقت بالثواني
            last_results = results

            print(f"Processed {len(results)} molecules in {processing_time:.3f} s")

    return render_template(
        "index.html",
        results=results,
        original_text=original_text,
        has_excel=results is not None,
        processing_time=processing_time
    )


# ===========================
# 🔥 Route لتحميل النتائج
# ===========================
@app.route("/download")
def download_results():
    global last_results
    if last_results is None:
        return "No results to download!", 400

    df = logic.results_to_dataframe(last_results)

    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")
    output.seek(0)

    return send_file(
        output,
        as_attachment=True,
        download_name="Pharmetix_results.xlsx",
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 7860))  # Render يعطي رقم بورت تلقائي
    print(f">>> Pharmetix server running on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False)
    