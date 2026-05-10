from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from fairshift.data import append_result, empty_results, format_prompt_for_income
from fairshift.llm_clients import build_clients, get_llm_response
from fairshift.parsing import extract_binary_response_lenient

INPUT_DATA = Path("data/sampled_datasets/1000_samples.csv")
OUTPUT_DIR = Path("data/results/scaling")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Gemma scaling experiments.")
    parser.add_argument("--model-name", default="Gemma27B", choices=["Gemma2B", "Gemma9B", "Gemma27B"])
    parser.add_argument("--input", default=str(INPUT_DATA))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clients = build_clients()
    df = pd.read_csv(args.input)
    df.columns = [
        "AGEP",
        "COW",
        "SCHL",
        "MAR",
        "OCCP",
        "POBP",
        "RELP",
        "WKHP",
        "SEX",
        "RAC1P",
        "label",
        "PINCP",
        "Year",
        "State",
    ]

    results = empty_results()
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing predictions"):
        prompt = format_prompt_for_income(row)
        raw_response = get_llm_response(prompt, args.model_name, clients)
        response = extract_binary_response_lenient(raw_response)
        append_result(results, row, args.model_name, response)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{args.model_name}_1000_samples.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
