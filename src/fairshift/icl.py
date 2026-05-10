from __future__ import annotations

import random

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from fairshift.data import format_prompt_for_income


class EnhancedCoverageSelector:
    """Coverage-based exemplar selection for ICL prompts."""

    def __init__(
        self,
        n_exemplars: int = 5,
        coverage_metric: str = "cosine",
        add_candidate_score: bool = True,
        candidate_score_discount: float = 1.0,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.n_exemplars = n_exemplars
        self.coverage_metric = coverage_metric
        self.add_candidate_score = add_candidate_score
        self.candidate_score_discount = candidate_score_discount
        self.encoder = SentenceTransformer(embedding_model)

    def compute_coverage_scores(self, query_embedding, candidate_embeddings) -> np.ndarray:
        if self.coverage_metric != "cosine":
            raise ValueError(f"Unsupported coverage metric: {self.coverage_metric}")
        return cosine_similarity(
            query_embedding.unsqueeze(0).cpu().numpy(),
            candidate_embeddings.cpu().numpy(),
        )[0]

    def select_exemplars(self, candidate_pool_df: pd.DataFrame, query: str | None = None) -> pd.DataFrame:
        prompts = candidate_pool_df.apply(
            lambda row: format_prompt_for_income(row, include_instruction=False), axis=1
        ).tolist()
        candidate_embeddings = self.encoder.encode(prompts, convert_to_tensor=True)

        if query:
            query_embedding = self.encoder.encode(query, convert_to_tensor=True)
            coverage_scores = self.compute_coverage_scores(query_embedding, candidate_embeddings)
        else:
            first_idx = random.randint(0, len(prompts) - 1)
            coverage_scores = self.compute_coverage_scores(candidate_embeddings[first_idx], candidate_embeddings)

        selected_indices: list[int] = []
        current_coverage = np.zeros_like(coverage_scores)

        while len(selected_indices) < self.n_exemplars:
            remaining_indices = list(set(range(len(prompts))) - set(selected_indices))
            if not remaining_indices:
                break

            coverage_gains = np.maximum(
                coverage_scores[remaining_indices] - current_coverage[remaining_indices], 0
            )
            if self.add_candidate_score:
                coverage_gains += coverage_scores[remaining_indices] / self.candidate_score_discount

            best_remaining_idx = remaining_indices[int(np.argmax(coverage_gains))]
            selected_indices.append(best_remaining_idx)
            current_coverage = np.maximum(current_coverage, coverage_scores[best_remaining_idx])

        return candidate_pool_df.iloc[selected_indices]


def create_icl_prompt(query: str, exemplars_df: pd.DataFrame) -> str:
    prompt = "Given the following examples of income predictions:\n\n"
    for _, exemplar in exemplars_df.iterrows():
        income_label = "Above" if exemplar["label"] == 1 else "Below"
        prompt += "Example:\n"
        prompt += f"Person: {format_prompt_for_income(exemplar, include_instruction=False)}\n"
        prompt += f"Income: {income_label}\n\n"

    prompt += f"Now, predict for this person:\n{query}\n"
    prompt += (
        "Based on this information, respond with only 'Above' or 'Below' to indicate "
        "if this individual's income is likely to be above $50,000 per year."
    )
    return prompt
