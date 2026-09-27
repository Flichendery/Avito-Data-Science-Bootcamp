# Avito-Data-Science-Bootcamp

Тестовое задание для Avito Bootcamp по NLP и LLM.

## 01_baseline

Простой retrieval baseline на основе word-level TF-IDF

Для каждого объявления объединяются `item_title_raw` и `item_description_raw`. Запрос и объявления переводятся в TF-IDF-векторы, после чего выбираются 50 объявлений с максимальной косинусной близостью

Baseline не использует категории, локацию, историю взаимодействий или морфологический анализ

## Ограничения

- Не учитывает словоформы и опечатки
- Не использует дополнительные признаки объявления
- Каждый запрос обрабатывается независимо

## Результат

Локальная валидация: **Recall@50 = 0.166923**

На Степике: **Recall@50 = 0.192578**

## Запуск
Установить зависиомсти
```bash
pip install -r requirements.txt
```
Разместить `train.parquet` `benchmark_queries.parquet` `benchmark_items.parquet` в корень проекта
```bash
python main.py
```
