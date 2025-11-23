from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, FilterCatalog, Crippen
from rdkit.Chem.Scaffolds import MurckoScaffold
import pandas as pd
import os, gzip, warnings

warnings.filterwarnings("ignore", message="to-Python converter for class boost::shared_ptr")

# ---------- CONFIG ----------
INPUT_FILE  = "natural.csv"      # للاستخدام الأوفلاين (من غير الويب)
OUTPUT_FILE = "ranked_results_.csv"
SCAFFOLD_LIMIT = 3
MAX_MOLS = 50_000
PRINT_EVERY = 5_000
# ----------------------------


# ========= تحميل الجزيئات من ملف (أوفلاين) ==========
def load_molecules(path, max_mols=None):
    ext = os.path.splitext(path)[-1].lower()
    mols = []

    if ext == ".smi":
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if max_mols and i >= max_mols:
                    break
                if not line.strip():
                    continue
                smi = line.split()[0]
                m = Chem.MolFromSmiles(smi)
                if m:
                    mols.append((smi, m))

    elif ext == ".csv":
        df = pd.read_csv(path)
        if "smiles" not in df.columns:
            raise ValueError("CSV file must have a header named 'smiles'")
        for i, smi in enumerate(df["smiles"]):
            if max_mols and i >= max_mols:
                break
            if not isinstance(smi, str) or not smi.strip():
                continue
            m = Chem.MolFromSmiles(smi)
            if m:
                mols.append((smi, m))

    elif ext == ".sdf":
        suppl = Chem.SDMolSupplier(path, sanitize=True, removeHs=True)
        for i, m in enumerate(suppl):
            if max_mols and i >= max_mols:
                break
            if m:
                smi = Chem.MolToSmiles(m)
                mols.append((smi, m))
            if PRINT_EVERY and (i + 1) % PRINT_EVERY == 0:
                print(f"  …read {i+1} molecules")

    elif ext == ".gz" and path.endswith(".sdf.gz"):
        with gzip.open(path, "rb") as fh:
            suppl = Chem.ForwardSDMolSupplier(fh, sanitize=True, removeHs=True)
            for i, m in enumerate(suppl):
                if max_mols and i >= max_mols:
                    break
                if m:
                    smi = Chem.MolToSmiles(m)
                    mols.append((smi, m))
                if PRINT_EVERY and (i + 1) % PRINT_EVERY == 0:
                    print(f"  …read {i+1} molecules")
    else:
        raise ValueError("File must be .smi, .csv, .sdf, or .sdf.gz")

    return mols


# ========= الفلاتر والقواعد الفيزيائية ==========
METALS = {
    3, 4, 11, 12, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    37, 38, 39, 40, 47, 48, 49, 50, 55, 56
}

def keep_organic_single(m):
    """نحتفظ فقط بجزيء عضوي واحد (بدون معادن، بدون disconnected)."""
    smi = Chem.MolToSmiles(m)
    if "." in smi:
        return None
    for a in m.GetAtoms():
        if a.GetAtomicNum() in METALS:
            return None
    return m


def make_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)

CATALOG = make_catalog()

def has_alerts(m):
    return CATALOG.HasMatch(m)


def lipinski(m):
    return (
        Descriptors.MolWt(m) <= 500 and
        Descriptors.MolLogP(m) <= 5 and
        rdMolDescriptors.CalcNumHBD(m) <= 5 and
        rdMolDescriptors.CalcNumHBA(m) <= 10
    )


def veber(m):
    return (
        rdMolDescriptors.CalcNumRotatableBonds(m) <= 10 and
        rdMolDescriptors.CalcTPSA(m) <= 140
    )


def ghose(m):
    mw    = Descriptors.MolWt(m)
    logp  = Descriptors.MolLogP(m)
    atoms = m.GetNumAtoms()
    try:
        mr = Crippen.MolMR(m)
    except Exception:
        mr = -1.0
    return (
        160 <= mw <= 480 and
        -0.4 <= logp <= 5.6 and
        20 <= atoms <= 70 and
        40 <= mr <= 130
    )


def egan(m):
    return (
        Descriptors.MolLogP(m) <= 5.88 and
        rdMolDescriptors.CalcTPSA(m) <= 131
    )


def muegge(m):
    return (
        200 <= Descriptors.MolWt(m) <= 600 and
        -2   <= Descriptors.MolLogP(m) <= 5 and
        rdMolDescriptors.CalcNumHBA(m) <= 10 and
        rdMolDescriptors.CalcNumHBD(m) <= 5 and
        rdMolDescriptors.CalcNumRotatableBonds(m) <= 15 and
        rdMolDescriptors.CalcTPSA(m) <= 150
    )


def bioavailability(m):
    """نفس المنطق: 0.55 إذا مرّ Lipinski + Veber، غير كذا 0.11."""
    return 0.55 if (lipinski(m) and veber(m)) else 0.11


def murcko(m):
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


# ========= أوفلاين: تحليل مكتبة كاملة من ملف وحفظ ranked_results_.csv ==========
def analyze_library(path,
                    out_csv=OUTPUT_FILE,
                    scaffold_limit=SCAFFOLD_LIMIT,
                    max_mols=MAX_MOLS):
    """
    نفس الفنكشن الأصلي حقك تقريبًا:
    - يقرأ مكتبة من ملف
    - يطبّق الفلاتر
    - يحسب السكور
    - يحفظ النتائج اللي "مرّت" فقط في ملف CSV
    """
    mols = load_molecules(path, max_mols=max_mols)
    print(f"📥 Loaded {len(mols)} molecules (limit={max_mols}) from {path}")

    results = []
    scaf_counts = {}

    for idx, (smi, m) in enumerate(mols, 1):
        m = keep_organic_single(m)
        if not m:
            continue
        if has_alerts(m):
            continue
        if not lipinski(m) or not veber(m):
            continue

        scaf = murcko(m)
        if scaf:
            cnt = scaf_counts.get(scaf, 0)
            if cnt >= scaffold_limit:
                continue
            scaf_counts[scaf] = cnt + 1

        # حساب الديسكربتورز:
        molwt = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        num_hba = rdMolDescriptors.CalcNumHBA(m)
        num_hbd = rdMolDescriptors.CalcNumHBD(m)
        num_rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        molar_refractivity = Crippen.MolMR(m)
        num_atoms = m.GetNumAtoms()

        g   = ghose(m)
        e   = egan(m)
        mu  = muegge(m)
        bio = bioavailability(m)
        score = bio + 0.1 * (int(g) + int(e) + int(mu))

        results.append({
            "SMILES": smi,
            "MolWt": molwt,
            "MolLogP": logp,
            "NumAtoms": num_atoms,
            "NumHBA": num_hba,
            "NumHBD": num_hbd,
            "NumRotatableBonds": num_rot,
            "TPSA": tpsa,
            "MolarRefractivity": molar_refractivity,
            "Lipinski": True,
            "Veber": True,
            "Ghose": g,
            "Egan": e,
            "Muegge": mu,
            "Bioavailability": bio,
            "Final_Score": round(score, 4),
            "Scaffold": scaf
        })

        if PRINT_EVERY and idx % PRINT_EVERY == 0:
            print(f"  …filtered {idx} / {len(mols)}")

    df = pd.DataFrame(results).sort_values(
        "Final_Score", ascending=False
    ).reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    print(f"✅ {len(df)} molecules passed → saved to {out_csv}")
    return df


# ========= أونلاين: الفنكشن اللي يستدعيه Flask (Pharmetix) ==========
def run_model_on_input(smiles_list,
                       scaffold_limit=SCAFFOLD_LIMIT):
    """
    هذا الفنكشن هو قلب الموقع:
    - يستقبل قائمة SMILES (من المربع + من ملف Excel/CSV)
    - يرجّع list of dicts فيها:
        input_text    -> الـ SMILES الأصلي
        drug_likeness -> Yes / No / Error
        details       -> سبب النتيجة + أهم الديسكربتورز
        (وباقي الأعمدة جاهزة للتصدير كـ Excel)
    """

    results = []
    scaf_counts = {}

    for smi in smiles_list:
        smi = str(smi).strip()
        if not smi:
            continue

        m = Chem.MolFromSmiles(smi)
        if not m:
            # SMILES غير صالح
            results.append({
                "input_text": smi,
                "drug_likeness": "Error",
                "details": "Invalid SMILES string.",
                "MolWt": None,
                "MolLogP": None,
                "NumAtoms": None,
                "NumHBA": None,
                "NumHBD": None,
                "NumRotatableBonds": None,
                "TPSA": None,
                "MolarRefractivity": None,
                "Lipinski": None,
                "Veber": None,
                "Ghose": None,
                "Egan": None,
                "Muegge": None,
                "Bioavailability": None,
                "Final_Score": None,
                "Scaffold": None,
            })
            continue

        # فلتر: عضوي، جزيء واحد، بدون معادن
        m2 = keep_organic_single(m)
        if not m2:
            results.append({
                "input_text": smi,
                "drug_likeness": "No",
                "details": "Non-organic / metal / disconnected species.",
                "MolWt": None,
                "MolLogP": None,
                "NumAtoms": None,
                "NumHBA": None,
                "NumHBD": None,
                "NumRotatableBonds": None,
                "TPSA": None,
                "MolarRefractivity": None,
                "Lipinski": False,
                "Veber": False,
                "Ghose": None,
                "Egan": None,
                "Muegge": None,
                "Bioavailability": None,
                "Final_Score": None,
                "Scaffold": None,
            })
            continue
        m = m2

        # فلتر PAINS / Brenk
        if has_alerts(m):
            results.append({
                "input_text": smi,
                "drug_likeness": "No",
                "details": "Fails PAINS/Brenk filter (structural alerts detected).",
                "MolWt": None,
                "MolLogP": None,
                "NumAtoms": None,
                "NumHBA": None,
                "NumHBD": None,
                "NumRotatableBonds": None,
                "TPSA": None,
                "MolarRefractivity": None,
                "Lipinski": False,
                "Veber": False,
                "Ghose": None,
                "Egan": None,
                "Muegge": None,
                "Bioavailability": None,
                "Final_Score": None,
                "Scaffold": None,
            })
            continue

        # قواعد Lipinski & Veber
        passed_lipinski = lipinski(m)
        passed_veber    = veber(m)
        if not (passed_lipinski and passed_veber):
            results.append({
                "input_text": smi,
                "drug_likeness": "No",
                "details": "Fails Lipinski / Veber rules.",
                "MolWt": None,
                "MolLogP": None,
                "NumAtoms": None,
                "NumHBA": None,
                "NumHBD": None,
                "NumRotatableBonds": None,
                "TPSA": None,
                "MolarRefractivity": None,
                "Lipinski": passed_lipinski,
                "Veber": passed_veber,
                "Ghose": None,
                "Egan": None,
                "Muegge": None,
                "Bioavailability": None,
                "Final_Score": None,
                "Scaffold": None,
            })
            continue

        # فلتر Murcko scaffold diversity
        scaf = murcko(m)
        if scaf:
            cnt = scaf_counts.get(scaf, 0)
            if cnt >= scaffold_limit:
                results.append({
                    "input_text": smi,
                    "drug_likeness": "No",
                    "details": "Scaffold over-represented (diversity filter).",
                    "MolWt": None,
                    "MolLogP": None,
                    "NumAtoms": None,
                    "NumHBA": None,
                    "NumHBD": None,
                    "NumRotatableBonds": None,
                    "TPSA": None,
                    "MolarRefractivity": None,
                    "Lipinski": True,
                    "Veber": True,
                    "Ghose": None,
                    "Egan": None,
                    "Muegge": None,
                    "Bioavailability": None,
                    "Final_Score": None,
                    "Scaffold": scaf,
                })
                continue
            scaf_counts[scaf] = cnt + 1

        # ========== هنا المركب "مرّ" كل الفلاتر ==========
        molwt = Descriptors.MolWt(m)
        logp = Descriptors.MolLogP(m)
        num_hba = rdMolDescriptors.CalcNumHBA(m)
        num_hbd = rdMolDescriptors.CalcNumHBD(m)
        num_rot = rdMolDescriptors.CalcNumRotatableBonds(m)
        tpsa = rdMolDescriptors.CalcTPSA(m)
        molar_refractivity = Crippen.MolMR(m)
        num_atoms = m.GetNumAtoms()

        g   = ghose(m)
        e   = egan(m)
        mu  = muegge(m)
        bio = bioavailability(m)
        score = bio + 0.1 * (int(g) + int(e) + int(mu))

        detail_str = (
            f"Score={score:.3f}; "
            f"Lipinski: True, Veber: True, Ghose: {g}, Egan: {e}, Muegge: {mu}; "
            f"MW={molwt:.1f}, logP={logp:.2f}, HBA={num_hba}, HBD={num_hbd}, "
            f"RotB={num_rot}, TPSA={tpsa:.1f}, MR={molar_refractivity:.1f}, "
            f"Atoms={num_atoms}, Scaffold={scaf}"
        )

        results.append({
            "input_text": smi,
            "drug_likeness": "Yes",
            "details": detail_str,
            "MolWt": molwt,
            "MolLogP": logp,
            "NumAtoms": num_atoms,
            "NumHBA": num_hba,
            "NumHBD": num_hbd,
            "NumRotatableBonds": num_rot,
            "TPSA": tpsa,
            "MolarRefractivity": molar_refractivity,
            "Lipinski": True,
            "Veber": True,
            "Ghose": g,
            "Egan": e,
            "Muegge": mu,
            "Bioavailability": bio,
            "Final_Score": round(score, 4),
            "Scaffold": scaf,
        })

    return results


# ========= Helper: تحويل نتائج الويب إلى DataFrame (للدَونلود) ==========
def results_to_dataframe(results):
    """
    يساعدك في Flask لما تحب تخلّي المستخدم يحمّل النتائج كـ CSV/Excel:
        df = results_to_dataframe(results)
        df.to_csv(...), أو df.to_excel(...)
    """
    return pd.DataFrame(results)


# تشغيل أوفلاين من غير الويب:
if __name__ == "__main__":
    df = analyze_library(INPUT_FILE)
    print(df.head())
