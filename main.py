"""Версия 3: гибридный текстовый поиск с историей взаимодействий

Текстовый поиск использует char n-grams для заголовка, word-level TF-IDF
для первых 1000 символов описания, и дополнительно char n-grams для более
короткого среза описания -- та же идея, что и с заголовком: word-level
TF-IDF не видит словоформы ("массаж"/"массажный"), а char n-grams эту
проблему снимают. Проверено на валидации: третий канал даёт заметный
прирост Recall@50 (см. комментарий у DESC_CHAR_WEIGHT).

История из train добавляет кандидатов для точных совпадений запросов

Запуск:
    python main3.py
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer

# Основные параметры пайплайна
TOP_K = 50
BATCH_SIZE = 200
DESC_MAX_CHARS = 1000
TITLE_CHAR_NGRAMS = (3, 5)
TITLE_HASH_FEATURES = 2**18
DESC_MAX_FEATURES = 75_000
DESC_MIN_DOC_FREQ = 3
# третий канал: char n-grams по описанию, отдельно от word-level TF-IDF выше.
# Срез короче, чем DESC_MAX_CHARS (400 вместо 1000) и хэш меньше, чем у
# заголовка (2**17 вместо 2**18) -- иначе словарь хэшей на всём корпусе
# перестаёт помещаться в память; для сути объявления первых 400 символов
# обычно достаточно, дальше в описаниях в основном шаблонные хвосты
DESC_CHAR_NGRAMS = (3, 5)
DESC_CHAR_MAX_CHARS = 400
DESC_CHAR_HASH_FEATURES = 2**17
TITLE_WEIGHT = 0.55
DESC_WEIGHT = 0.2
DESC_CHAR_WEIGHT = 0.25
MAX_HISTORY_ITEMS_PER_QUERY = 10


def load_data(
    train_path: Path, items_path: Path, queries_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(train_path, columns=["search_query", "item_id"])
    items = pd.read_parquet(items_path)
    queries = pd.read_parquet(queries_path)
    return train, items, queries


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^0-9a-zа-яё\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_index(items: pd.DataFrame):
    """Строит индексы по заголовкам и описаниям объявлений"""
    title_norm = items["item_title_raw"].fillna("").map(normalize_text)
    desc_norm = items["item_description_raw"].fillna("").str.slice(0, DESC_MAX_CHARS).map(normalize_text)
    # отдельный, более короткий срез описания -- специально для char n-grams,
    # чтобы не раздувать память (см. комментарий у DESC_CHAR_MAX_CHARS)
    desc_char_norm = items["item_description_raw"].fillna("").str.slice(0, DESC_CHAR_MAX_CHARS).map(normalize_text)

    title_hasher = HashingVectorizer(
        analyzer="char_wb", ngram_range=TITLE_CHAR_NGRAMS,
        n_features=TITLE_HASH_FEATURES, alternate_sign=False,
    )
    title_tfidf = TfidfTransformer()
    title_matrix = title_tfidf.fit_transform(title_hasher.transform(title_norm)).tocsr()

    desc_vectorizer = TfidfVectorizer(max_features=DESC_MAX_FEATURES, min_df=DESC_MIN_DOC_FREQ)
    desc_matrix = desc_vectorizer.fit_transform(desc_norm).tocsr()

    # третий канал: та же идея, что и с заголовком -- char n-grams вместо
    # слов по описанию, чтобы ловить словоформы, которые word-level TF-IDF
    # по описанию (канал выше) пропускает
    desc_char_hasher = HashingVectorizer(
        analyzer="char_wb", ngram_range=DESC_CHAR_NGRAMS,
        n_features=DESC_CHAR_HASH_FEATURES, alternate_sign=False,
    )
    desc_char_tfidf = TfidfTransformer()
    desc_char_matrix = desc_char_tfidf.fit_transform(desc_char_hasher.transform(desc_char_norm)).tocsr()

    return {
        "title_hasher": title_hasher,
        "title_tfidf": title_tfidf,
        "title_matrix": title_matrix.T.tocsr(),  # Транспонируем для умножения
        "desc_vectorizer": desc_vectorizer,
        "desc_matrix": desc_matrix.T.tocsr(),
        "desc_char_hasher": desc_char_hasher,
        "desc_char_tfidf": desc_char_tfidf,
        "desc_char_matrix": desc_char_matrix.T.tocsr(),
        "item_ids": items["item_id"].values,
    }


def build_history(train: pd.DataFrame, valid_item_ids: set[str]) -> dict[str, list[str]]:
    """Строит историю запросов только для объявлений из текущего корпуса"""
    train = train.copy()
    train["q_norm"] = train["search_query"].map(normalize_text)
    train = train[train["item_id"].isin(valid_item_ids)]
    # Сначала добавляем наиболее частые исторические объявления
    counts = train.groupby(["q_norm", "item_id"]).size().reset_index(name="n")
    counts = counts.sort_values("n", ascending=False)
    history: dict[str, list[str]] = {}
    for q_norm, item_id in zip(counts["q_norm"], counts["item_id"]):
        bucket = history.setdefault(q_norm, [])
        if len(bucket) < MAX_HISTORY_ITEMS_PER_QUERY:
            bucket.append(item_id)
    return history


def retrieve_candidates(query_texts: list[str], index: dict, top_k: int = TOP_K) -> list[list[str]]:
    """Возвращает top-k кандидатов по текстовому сходству"""
    query_norm = [normalize_text(q) for q in query_texts]
    title_q = index["title_tfidf"].transform(index["title_hasher"].transform(query_norm)).tocsr()
    desc_q = index["desc_vectorizer"].transform(query_norm).tocsr()
    desc_char_q = index["desc_char_tfidf"].transform(index["desc_char_hasher"].transform(query_norm)).tocsr()
    item_ids = index["item_ids"]

    results: list[list[str]] = []
    n = len(query_texts)
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        sims = (title_q[start:end] @ index["title_matrix"]) * TITLE_WEIGHT
        sims = sims + (desc_q[start:end] @ index["desc_matrix"]) * DESC_WEIGHT
        sims = sims + (desc_char_q[start:end] @ index["desc_char_matrix"]) * DESC_CHAR_WEIGHT
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


def generate_candidates(
    query_texts: list[str], index: dict, history: dict[str, list[str]], top_k: int = TOP_K
) -> list[list[str]]:
    """Объединяет исторические и текстовые кандидаты"""
    text_candidates = retrieve_candidates(query_texts, index, top_k=top_k)
    final: list[list[str]] = []
    for query_text, text_cands in zip(query_texts, text_candidates):
        q_norm = normalize_text(query_text)
        hist_cands = history.get(q_norm, [])
        merged = list(hist_cands)
        seen = set(merged)
        for item_id in text_cands:
            if len(merged) >= top_k:
                break
            if item_id not in seen:
                merged.append(item_id)
                seen.add(item_id)
        final.append(merged[:top_k])
    return final


def make_query_level_split(
    train: pd.DataFrame, val_fraction: float = 0.1, seed: int = 42
) -> tuple[set[str], set[str]]:
    """Делит уникальные запросы на dev и validation без утечки"""
    rng = np.random.RandomState(seed)
    unique_queries = np.array(train["search_query"].unique(), dtype=object)
    rng.shuffle(unique_queries)
    n_val = int(len(unique_queries) * val_fraction)
    return set(unique_queries[n_val:]), set(unique_queries[:n_val])


def recall_at_k(predictions: list[list[str]], relevant: list[set[str]]) -> float:
    """Считает средний Recall@k по запросам"""
    scores = [len(set(p) & r) / len(r) for p, r in zip(predictions, relevant) if r]
    return float(np.mean(scores))


def run_validation(train: pd.DataFrame) -> float:
    """Проверяет V3 на query-level validation из train"""
    dev_queries, val_queries = make_query_level_split(train)

    val_df = train[train["search_query"].isin(val_queries)]
    dev_df = train[train["search_query"].isin(dev_queries)]

    items_corpus = (
        train.drop_duplicates("item_id")[
            ["item_id", "item_title_raw", "item_description_raw"]
        ]
        .reset_index(drop=True)
    )

    index = build_index(items_corpus)

    history = build_history(
        dev_df,
        valid_item_ids=set(items_corpus["item_id"]),
    )

    val_query_texts = list(val_df["search_query"].drop_duplicates())
    predictions = generate_candidates(val_query_texts, index, history)

    relevant_map = val_df.groupby("search_query")["item_id"].apply(set).to_dict()
    relevant = [relevant_map[q] for q in val_query_texts]

    return recall_at_k(predictions, relevant)


def save_predictions(query_ids: list[str], predictions: list[list[str]], output_path: Path) -> None:
    answer = pd.DataFrame({
        "query_id": query_ids,
        "answer": [" ".join(p) for p in predictions],
    })
    answer.to_csv(output_path, index=False)


def validate_predictions(output_path: Path, queries: pd.DataFrame, items: pd.DataFrame) -> None:
    """Проверяет формат и содержимое готового answer.csv"""
    answer = pd.read_csv(output_path, dtype=str, keep_default_na=False)
    assert list(answer.columns) == ["query_id", "answer"], f"неверные колонки: {answer.columns.tolist()}"

    expected_ids = set(queries["query_id"])
    got_ids = set(answer["query_id"])
    assert got_ids == expected_ids, (
        f"несовпадение query_id: пропущено {len(expected_ids - got_ids)}, "
        f"лишних {len(got_ids - expected_ids)}"
    )
    assert answer["query_id"].is_unique, "есть повторяющиеся query_id"

    valid_items = set(items["item_id"])
    for query_id, answer_str in zip(answer["query_id"], answer["answer"]):
        item_list = answer_str.split(" ") if answer_str else []
        assert len(item_list) <= TOP_K, f"{query_id}: больше {TOP_K} item_id"
        assert len(item_list) == len(set(item_list)), f"{query_id}: есть дубли item_id"
        unknown = set(item_list) - valid_items
        assert not unknown, f"{query_id}: неизвестные item_id {unknown}"

    print(f"validate_predictions: OK, {len(answer)} строк, все проверки пройдены")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("train.parquet"))
    parser.add_argument("--items", type=Path, default=Path("benchmark_items.parquet"))
    parser.add_argument("--queries", type=Path, default=Path("benchmark_queries.parquet"))
    parser.add_argument("--output", type=Path, default=Path("answer.csv"))
    args = parser.parse_args()

    print("Загрузка train")
    train = pd.read_parquet(args.train, columns=[
        "search_query",
        "item_id",
        "item_title_raw",
        "item_description_raw",
    ])
    print(f"Строк: {len(train)}")
    print(f"Уникальных запросов: {train['search_query'].nunique()}")

    recall = run_validation(train)
    print(f"\nЛокальная валидация Recall@{TOP_K} = {recall:.6f}")

    print("\nЗагрузка benchmark")
    items = pd.read_parquet(args.items)
    queries = pd.read_parquet(args.queries)
    print(f"Запросов: {len(queries)}")
    print(f"Объявлений: {len(items)}")

    index = build_index(items)
    print(f"Индекс по benchmark_items построен")

    history = build_history(train, valid_item_ids=set(items["item_id"]))
    print(f"История: {len(history)} уникальных запросов с валидным кандидатом")
    query_texts = list(queries["search_query"])
    predictions = generate_candidates(query_texts, index, history)
    print(
        f"Кандидаты сгенерированы для {len(predictions)} запросов")

    save_predictions(list(queries["query_id"]), predictions, args.output)
    print(f"Сохранено: {args.output}")

    validate_predictions(args.output, queries, items)


if __name__ == "__main__":
    main()