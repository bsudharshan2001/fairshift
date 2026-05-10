from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from fairshift.data import append_result, empty_results, format_prompt_for_income, load_state_data, set_seed
from fairshift.llm_clients import build_clients, get_llm_response
from fairshift.parsing import extract_binary_response

YEARS = [2014, 2016, 2018]
STATES = ["CA", "TX", "MI"]
MODELS = ["ChatGPT", "Claude", "Mistral"]
OUTPUT_DIR = Path("data/results/pre_icl")


def save_progress(year: int, state: str, results: dict[str, list]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(f"checkpoint_{year}_{state}.json").write_text(
        json.dumps({"year": year, "state": state, "timestamp": timestamp}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(results).to_csv(OUTPUT_DIR / f"results_{year}_{state}_{timestamp}.csv", index=False)


def main() -> None:
    set_seed()
    clients = build_clients()
    results = empty_results()
    progress_counter = 0

    for year in YEARS:
        for state in STATES:
            print(f"Loading data for {state} {year}...")
            df_sampled = load_state_data(state, year, sample_size=1000)
            prompts = df_sampled.apply(format_prompt_for_income, axis=1)

            for idx, prompt in enumerate(prompts):
                row = df_sampled.loc[idx]
                for model_name in MODELS:
                    raw_response = get_llm_response(prompt, model_name, clients)
                    response = extract_binary_response(raw_response)
                    append_result(results, row, model_name, response, year=year, state=state)
                    progress_counter += 1

                if progress_counter % 100 == 0:
                    save_progress(year, state, results)

            save_progress(year, state, results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "llm_income_prediction_analysis_1000.csv", index=False)
    print("Results saved successfully.")


if __name__ == "__main__":
    main()
