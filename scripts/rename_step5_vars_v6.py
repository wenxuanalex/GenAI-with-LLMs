import json
from pathlib import Path


NOTEBOOK_PATH = Path(r"C:\Users\jeeey\applied-machine-learning\Olist_DataPreparationv6.ipynb")


def to_lines(text: str):
    return [line + "\n" for line in text.strip("\n").split("\n")]


REPLACEMENTS = [
    ("step10_best_model_label", "step5_best_model_label"),
    ("step10_best_baseline_model_label", "step5_best_baseline_model_label"),
    ("step10_best_baseline_mae", "step5_best_baseline_mae"),
    ("step10_winner_label", "step5_winner_label"),
    ("step10_winner_mae", "step5_winner_mae"),
    ("y_test_step10", "y_test_step5"),
    ("best_step10", "best_step5"),
    ("y_pred_step10_strat", "y_pred_step5"),
    ("r2_step10_strat", "r2_step5"),
    ("rmse_step10_strat", "rmse_step5"),
    ("mae_step10_strat", "mae_step5"),
    ("build_step10_matrix", "build_step5_matrix"),
]


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    changed = False

    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))
        new_src = src
        for old, new in REPLACEMENTS:
            new_src = new_src.replace(old, new)
        if new_src != src:
            cell["source"] = to_lines(new_src)
            changed = True

    if changed:
        NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
        print("Renamed Step 5 variables for readability.")
    else:
        print("No Step 5 variable renames were needed.")


if __name__ == "__main__":
    main()
