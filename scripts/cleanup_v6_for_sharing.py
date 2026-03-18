import json
from pathlib import Path


NOTEBOOK_PATH = Path(r"C:\Users\jeeey\applied-machine-learning\Olist_DataPreparationv6.ipynb")


def to_lines(text: str):
    return [line + "\n" for line in text.strip("\n").split("\n")]


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Remove empty code cells
    cells[:] = [
        cell for cell in cells
        if not (cell.get("cell_type") == "code" and not "".join(cell.get("source", [])).strip())
    ]

    markdown_fixes = {
        15: """---
## Step 0: Baseline Model (Chronological Split, No Leakage)

**Goal:** Establish a simple baseline using only `distance_km` to predict `actual_delivery_days`, with a **chronological** train/test split (production-like).
""",
        20: """---
## Step 1: MLR with Seller Aggregates (Chronological Split, No Leakage)

**Goal:** Add engineered features and point-in-time seller aggregates while preserving chronological split integrity.
""",
        21: """---
## Step 2: Log-Transformation (Handling Skewness)

**Why:** The target variable (`actual_delivery_days`) is strongly right-skewed, so `log1p` helps linear models fit the distribution more stably before predictions are transformed back to the original scale.
""",
        24: """---
## Step 3: Target Encoding (Cardinality Management)

**Why:** High-cardinality categorical variables are encoded into compact numeric signals so the model can retain useful seller/customer/location information without exploding the feature space.
""",
        27: """---
## Step 4: Tree Models On The Chronological Split

**Goal:** Compare stronger non-linear models on the same chronological feature set before any stricter shortlist revalidation is introduced in Step 5.
""",
    }
    for idx, text in markdown_fixes.items():
        if idx < len(cells) and cells[idx].get("cell_type") == "markdown":
            cells[idx]["source"] = to_lines(text)

    replacements = [
        ("step10_best_model_label", "step5_best_model_label"),
        ("step10_best_baseline_model_label", "step5_best_baseline_model_label"),
        ("step10_best_baseline_mae", "step5_best_baseline_mae"),
        ("step10_winner_label", "step5_winner_label"),
        ("step10_winner_mae", "step5_winner_mae"),
        ("step10_shortlist", "step5_shortlist"),
        ("step10_results", "step5_results"),
        ("best_step10", "best_step5"),
    ]

    for cell in cells:
        src = "".join(cell.get("source", []))
        new_src = src
        for old, new in replacements:
            new_src = new_src.replace(old, new)
        if new_src != src:
            cell["source"] = to_lines(new_src)

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Cleaned up v6 notebook for sharing.")


if __name__ == "__main__":
    main()
