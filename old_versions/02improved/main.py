"""Версия 2: char n-grams для заголовка + word-level TF-IDF для описания

Заголовок и описание индексируются отдельно, затем их сходства объединяются с весами 0.8 и 0.2

Локальная валидация делится по уникальным поисковым запросам

Запуск:
    Разместить `train.parquet`, `benchmark_queries.parquet` и `benchmark_items.parquet` в корень проекта
    `python main2.py`
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer

TOP_K = 50
VAL_FRACTION = 0.1
RANDOM_SEED = 42
BATCH_SIZE = 200
DESC_MAX_CHARS = 400
TITLE_CHAR_NGRAMS = (3, 5)
TITLE_HASH_FEATURES = 2**18
DESC_MAX_FEATURES = 50_000
DESC_MIN_DOC_FREQ = 3
TITLE_WEIGHT = 0.8
DESC_WEIGHT = 0.2


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


def build_index(items_corpus: pd.DataFrame):
    """Строит два независимых индекса — по заголовку (char n-grams) и по
    описанию (word-level) — их потом объединяем взвешенной суммой сходств."""
    title_norm = items_corpus["item_title_raw"].fillna("").map(normalize_text)
    desc_norm = (
        items_corpus["item_description_raw"].fillna("").str.slice(0, DESC_MAX_CHARS).map(normalize_text)
    )

    title_hasher = HashingVectorizer(
        analyzer="char_wb", ngram_range=TITLE_CHAR_NGRAMS,
        n_features=TITLE_HASH_FEATURES, alternate_sign=False,
    )
    title_tfidf = TfidfTransformer()
    title_matrix = title_tfidf.fit_transform(title_hasher.transform(title_norm)).tocsr()

    desc_vectorizer = TfidfVectorizer(max_features=DESC_MAX_FEATURES, min_df=DESC_MIN_DOC_FREQ)
    desc_matrix = desc_vectorizer.fit_transform(desc_norm).tocsr()

    return (title_hasher, title_tfidf, title_matrix), (desc_vectorizer, desc_matrix)


def retrieve_candidates(
    query_texts: list[str],
    title_index,
    desc_index,
    item_ids: np.ndarray,
    top_k: int = TOP_K,
) -> list[list[str]]:
    """Возвращает top_k объявлений для каждого поискового запроса"""
    title_hasher, title_tfidf, title_matrix = title_index
    desc_vectorizer, desc_matrix = desc_index

    query_norm = [normalize_text(q) for q in query_texts]
    title_q = title_tfidf.transform(title_hasher.transform(query_norm)).tocsr()
    desc_q = desc_vectorizer.transform(query_norm).tocsr()

    title_matrix_t = title_matrix.T.tocsr()
    desc_matrix_t = desc_matrix.T.tocsr()

    results: list[list[str]] = []
    n = len(query_texts)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        sims = (title_q[start:end] @ title_matrix_t) * TITLE_WEIGHT
        sims = sims + (desc_q[start:end] @ desc_matrix_t) * DESC_WEIGHT
        sims = sims.tocsr()
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
    scores = [len(set(p) & r) / len(r) for p, r in zip(predictions, relevant) if r]
    return float(np.mean(scores))


def run_validation(train: pd.DataFrame) -> float:
    dev_queries, val_queries = make_query_level_split(train, VAL_FRACTION, RANDOM_SEED)
    val_df = train[train["search_query"].isin(val_queries)]
    val_relevant_map = val_df.groupby("search_query")["item_id"].apply(set).to_dict()

    items_corpus = (
        train.drop_duplicates("item_id")[["item_id", "item_title_raw", "item_description_raw"]]
        .reset_index(drop=True)
    )
    item_ids = items_corpus["item_id"].values

    title_index, desc_index = build_index(items_corpus)

    val_query_texts = list(val_relevant_map.keys())
    predictions = retrieve_candidates(val_query_texts, title_index, desc_index, item_ids)
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

    print("\nTF-IDF индексы")
    title_index, desc_index = build_index(items)
    item_ids = items["item_id"].astype(str).values

    print(f"top-{TOP_K} кандидатов")
    predictions = retrieve_candidates(
        queries["search_query"].fillna("").tolist(),
        title_index,
        desc_index,
        item_ids,
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
    print("Загрузка train")
    train = load_data(train_path)
    print(f"Строк: {len(train)}\nУникальных запросов: {train['search_query'].nunique()}")

    recall = run_validation(train)
    print(
        f"\nЛокальная валидация Recall@{TOP_K} = {recall:.6f}"
    )

    generate_answer()


if __name__ == "__main__":
    main()
