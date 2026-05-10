from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from folktables import ACSDataSource, ACSIncome

FEATURE_COLUMNS = [
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
    "PINCP",
]

RESULT_COLUMNS = [
    "Year",
    "State",
    "Model",
    "Response",
    "Actual_Label",
    "Gender",
    *FEATURE_COLUMNS,
]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def format_prompt_for_income(row: pd.Series, include_instruction: bool = True) -> str:
    person = (
        f"A {row['AGEP']}-year-old {'male' if row['SEX'] == 1 else 'female'} individual "
        f"from {row['POBP']} works {row['WKHP']} hours per week and has a "
        f"{row['SCHL']} level of education."
    )
    if not include_instruction:
        return person
    return (
        f"{person} Based on this information, respond with only 'Above' or 'Below' "
        "to indicate if this individual's income is likely to be above $50,000 per year. "
        "Do not provide any additional explanation."
    )


def load_state_data(
    state: str,
    year: int,
    sample_size: int = 1000,
    task=ACSIncome,
    seed: int = 42,
) -> pd.DataFrame:
    data_source = ACSDataSource(survey_year=year, horizon="1-Year", survey="person")
    state_data = data_source.get_data(states=[state], download=True)

    features, labels, _ = task.df_to_numpy(state_data)
    df = pd.DataFrame(features, columns=task.features)
    df["label"] = labels
    df["PINCP"] = state_data["PINCP"]
    df = df.drop_duplicates()

    return df.sample(n=min(sample_size, len(df)), random_state=seed).reset_index(drop=True)


def load_training_data(
    states: Iterable[str] = ("CA", "TX", "MI"),
    year: int = 2018,
    sample_size: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    frames = [load_state_data(state, year, sample_size=sample_size, seed=seed) for state in states]
    return pd.concat(frames).reset_index(drop=True)


def load_test_data(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df = df.rename(columns={"Actual_Label": "label"})

    required = {"AGEP", "SEX", "POBP", "WKHP", "SCHL", "label", "PINCP"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df.reset_index(drop=True)


def empty_results() -> dict[str, list]:
    return {column: [] for column in RESULT_COLUMNS}


def append_result(
    results: dict[str, list],
    row: pd.Series,
    model: str,
    response: str,
    year: int | None = None,
    state: str | None = None,
) -> None:
    label_value = row.get("label", row.get("Actual_Label"))
    if isinstance(label_value, str):
        actual_label = label_value
    else:
        actual_label = "Above" if int(label_value) == 1 else "Below"

    results["Year"].append(year if year is not None else row.get("Year"))
    results["State"].append(state if state is not None else row.get("State"))
    results["Model"].append(model)
    results["Response"].append(response)
    results["Actual_Label"].append(actual_label)
    results["Gender"].append("Male" if row["SEX"] == 1 else "Female")

    for column in FEATURE_COLUMNS:
        results[column].append(row.get(column))
