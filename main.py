"""Version 1: простой baseline на TF-IDF
Использует заголовок и описание объявления, а затем ищет top-50 по косинусной близости
Локальная валидация делится по уникальным поисковым запросам, чтобы избежать утечки
Запуск:
    Разместить `train.parquet` `benchmark_queries.parquet` `benchmark_items.parquet` в корень проекта
    `python main.py` 
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

TOP_K = 50
VAL_FRACTION = 0.1
RANDOM_SEED = 42
MAX_TFIDF_FEATURES = 50_000
MIN_DOC_FREQ = 3
BATCH_SIZE = 200  # Размер батча ограничивает потребление памяти


def load_data(train_path: Path) -> pd.DataFrame:
    return pd.read_parquet(train_path)


def normalize_text(text: str) -> str:
    """Приводит текст к нижнему регистру и убирает лишние символы"""
    text = text.lower()
    text = re.sub(r"[^0-9a-zа-яё\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_query_level_split(train: pd.DataFrame, val_fraction: float, seed: int):
    """Делит уникальные запросы на dev и validation без утечки между ними"""
    rng = np.random.RandomState(seed)
    unique_queries = np.array(train["search_query"].unique(), dtype=object)
    rng.shuffle(unique_queries)
    n_val = int(len(unique_queries) * val_fraction)
    val_queries = set(unique_queries[:n_val])
    dev_queries = set(unique_queries[n_val:])
    return dev_queries, val_queries


def build_index(corpus_text: pd.Series) -> tuple[TfidfVectorizer, csr_matrix]:
    vectorizer = TfidfVectorizer(max_features=MAX_TFIDF_FEATURES, min_df=MIN_DOC_FREQ)
    matrix = vectorizer.fit_transform(corpus_text).tocsr()
    return vectorizer, matrix


def retrieve_candidates(
    query_matrix, item_matrix, item_ids: np.ndarray, top_k: int = TOP_K
) -> list[list[str]]:
    """Возвращает top_k объявлений для каждого поискового запроса"""
    item_matrix_t = item_matrix.T.tocsr()
    results: list[list[str]] = []
    n = query_matrix.shape[0]
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        sims = (query_matrix[start:end] @ item_matrix_t).tocsr()
        for local_i in range(end - start):
            row = sims.getrow(local_i)
            idx, data = row.indices, row.data
            if len(idx) > top_k:
                top_local = np.argpartition(-data, top_k)[:top_k]
                idx = idx[top_local]
            else:
                idx = idx[np.argsort(-data)]
            results.append(list(item_ids[idx]))
    return results


def recall_at_k(predictions: list[list[str]], relevant: list[set[str]]) -> float:
    scores = []
    for pred, rel in zip(predictions, relevant):
        if not rel:
            continue
        scores.append(len(set(pred) & rel) / len(rel))
    return float(np.mean(scores))


def run_validation(train: pd.DataFrame) -> float:
    dev_queries, val_queries = make_query_level_split(train, VAL_FRACTION, RANDOM_SEED)
    val_df = train[train["search_query"].isin(val_queries)]
    val_relevant_map = val_df.groupby("search_query")["item_id"].apply(set).to_dict()

    items_corpus = (
        train.drop_duplicates("item_id")[["item_id", "item_title_raw", "item_description_raw"]]
        .reset_index(drop=True)
    )
    corpus_text = (
        items_corpus["item_title_raw"].fillna("") + " " + items_corpus["item_description_raw"].fillna("")
    ).map(normalize_text)

    vectorizer, item_matrix = build_index(corpus_text)
    item_ids = items_corpus["item_id"].values

    val_query_texts = list(val_relevant_map.keys())
    val_query_matrix = vectorizer.transform([normalize_text(q) for q in val_query_texts])

    predictions = retrieve_candidates(val_query_matrix, item_matrix, item_ids)
    relevant = [val_relevant_map[q] for q in val_query_texts]
    return recall_at_k(predictions, relevant)


def generate_answer() -> None:
    queries_path = Path("benchmark_queries.parquet")
    items_path = Path("benchmark_items.parquet")
    output_path = Path("answer.csv")

    print("\nЗагрузка benchmark")
    queries = pd.read_parquet(queries_path)
    items = pd.read_parquet(items_path)

    print(f"Запросов: {len(queries)}")
    print(f"Объявлений: {len(items)}")

    item_texts = (
        items["item_title_raw"].fillna("")
        + " "
        + items["item_description_raw"].fillna("")
    ).map(normalize_text)

    print("\nTF-IDF индекс")
    vectorizer, item_matrix = build_index(item_texts)

    query_texts = queries["search_query"].fillna("").map(normalize_text)
    query_matrix = vectorizer.transform(query_texts)

    print(f"top-{TOP_K} кандидатов")
    predictions = retrieve_candidates(
        query_matrix,
        item_matrix,
        items["item_id"].astype(str).values,
        top_k=TOP_K,
    )

    answers = pd.DataFrame(
        {
            "query_id": queries["query_id"].astype(str),
            "answer": [" ".join(item_ids) for item_ids in predictions],
        }
    )

    answers.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Сохранено в {output_path}")


def main() -> None:
    train_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("train.parquet")
    print(f"Загрузка train")
    train = load_data(train_path)
    print(f"Строк: {len(train)}\nУникальных запросов: {train['search_query'].nunique()}")

    recall = run_validation(train)
    print(f"\nЛокальная валидация Recall@{TOP_K} = {recall:.6f}")

    generate_answer()


if __name__ == "__main__":
    main()