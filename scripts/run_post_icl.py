from __future__ import annotations

from pathlib import Path

import pandas as pd

from fairshift.data import append_result, empty_results, format_prompt_for_income, load_test_data, load_training_data, set_seed
from fairshift.icl import EnhancedCoverageSelector, create_icl_prompt
from fairshift.llm_clients import build_clients, get_llm_response
from fairshift.parsing import extract_binary_response

TEST_DATA = Path("data/original_1000_samples.csv")
OUTPUT_DIR = Path("data/results/post_icl")


def main() -> None:
    set_seed()
    clients = build_clients()
    selector = EnhancedCoverageSelector(n_exemplars=5)

    print("Loading training data...")
    df_train = load_training_data(sample_size=1000)
    print(f"Training set size: {len(df_train)}")

    print("Loading test data...")
    df_test = load_test_data(TEST_DATA)
    print(f"Test set size: {len(df_test)}")

    print("Selecting exemplars...")
    exemplars_df = selector.select_exemplars(df_train)
    results = empty_results()

    for idx, row in df_test.iterrows():
        query = format_prompt_for_income(row, include_instruction=False)
        icl_prompt = create_icl_prompt(query, exemplars_df)

        for model_name in ["ChatGPT", "Claude", "Mistral"]:
            provider = "local" if model_name == "Mistral" else "api"
            raw_response = get_llm_response(
                icl_prompt,
                model_name,
                clients,
                provider=provider,
                max_tokens=1500 if provider == "local" else 50,
            )
            response = extract_binary_response(raw_response)
            append_result(results, row, model_name, response)

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(df_test)} samples")
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(OUTPUT_DIR / "llm_income_prediction_intermediate.csv", index=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "llm_income_prediction_final.csv", index=False)
    print("Results saved successfully.")


if __name__ == "__main__":
    main()
