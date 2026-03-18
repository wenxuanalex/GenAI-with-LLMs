import json
from pathlib import Path


NOTEBOOK_PATH = Path(r"C:\Users\jeeey\applied-machine-learning\Olist_DataPreparationv6.ipynb")


def lines(text: str):
    return [line + "\n" for line in text.strip("\n").split("\n")]


STEP5_MD = """---
## Step 5: Final Chronological Revalidation On Top 3 MAE Models
**Goal**: Use the Step 4 chronological screening table as a point-in-time shortlist, select the top 3 supported models by lowest `MAE`, and rerun only those models under stricter conditions: chronological split, point-in-time seller history, `TimeSeriesSplit`, out-of-fold target encoding for high-risk columns, and region-aware weighting.
"""


STEP5_KPI_MD = """### Step 6A: MAE Impact Summary (Step 4 Screening Winner vs Step 5 Winner)
Use the `mae_comparison_v5` table above as the final shortlist revalidation check.

- Step 4 screening baseline: `step10_best_baseline_model_label`
- Step 5 winner: `step10_best_model_label`
- Impact formula: `Delta_MAE = mae_step10_strat - step10_best_baseline_mae`
- Interpretation: `Delta_MAE < 0` indicates the stricter Step 5 framework improved on the original Step 4 winner
"""


def main():
    nb = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    cells = nb["cells"]

    # Step 4 summary should stay a screening-only table.
    final_src = "".join(cells[40]["source"])
    screening_only_line = 'step10_label = clean_model_label(_get(g, "step10_best_model_label", "Step 5 shortlisted winner"))\nadd_row(rows_chrono, step10_label, "r2_step10_strat", "rmse_step10_strat", "mae_step10_strat", "CHRONO", baseline_rmse=baseline_rmse_chrono, baseline_mae=baseline_mae_chrono)\n'
    if screening_only_line in final_src:
        final_src = final_src.replace(screening_only_line, "")
        cells[40]["source"] = lines(final_src)

    # Make Step 5 wording explicitly depend on Step 4 screening.
    cells[42]["source"] = lines(STEP5_MD)
    cells[48]["source"] = lines(STEP5_KPI_MD)

    # Rename the Step 5 results dataframe for clarity in the code.
    step5_src = "".join(cells[43]["source"])
    replacements = [
        ('step10_results_df = pd.DataFrame(step10_results).sort_values(["mae", "rmse"]).reset_index(drop=True)', 'step5_revalidation_df = pd.DataFrame(step10_results).sort_values(["mae", "rmse"]).reset_index(drop=True)'),
        ('display(step10_results_df)', 'print("[Step 5] Shortlist revalidation results")\ndisplay(step5_revalidation_df)'),
        ('best_step10 = step10_results_df.iloc[0]', 'best_step10 = step5_revalidation_df.iloc[0]'),
        ('step10_best_model_label = f"{best_step10[\'baseline_model\']} under Step 10"', 'step10_best_model_label = f"{best_step10[\'baseline_model\']} under Step 5"'),
        ('[Step 10] Winner:', '[Step 5] Winner:'),
        ('[Step 10] Baseline screening winner was', '[Step 5] Baseline screening winner was'),
    ]
    for old, new in replacements:
        step5_src = step5_src.replace(old, new)
    cells[43]["source"] = lines(step5_src)

    kpi_src = "".join(cells[47]["source"])
    kpi_src = kpi_src.replace(
        '# MAE comparison table: screening winner vs Step 5 winner',
        '# MAE comparison table: Step 4 screening winner vs Step 5 winner'
    )
    cells[47]["source"] = lines(kpi_src)

    NOTEBOOK_PATH.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Separated Step 4 screening summary from Step 5 shortlist revalidation in v6 notebook.")


if __name__ == "__main__":
    main()
