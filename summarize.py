import os
import json
import sys
import re
from pathlib import Path
from llama_cpp import Llama
from tqdm.auto import tqdm


# ===============================
# ЗАГРУЗКА ТРАНСКРИПЦИИ
# ===============================
def get_transcription_text(file_path: str) -> str:
    """
    Преобразует исходную транскрибацию (JSON или TXT)
    в «плоский» текст для последующей суммаризации.
    """
    if file_path.endswith(".json"):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return "\n".join(seg["text"] for seg in data if seg.get("text", "").strip())

    elif file_path.endswith(".txt"):
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        texts = []
        for line in lines:
            # Обрезаем таймкоды вида: [00:00:01.000 - 00:00:05.000] Текст...
            if "] " in line:
                texts.append(line.split("] ", 1)[1])
            else:
                texts.append(line)
        return "\n".join(texts)

    else:
        raise ValueError("Only .txt and .json supported")


# ===============================
# ЧАНКИНГ
# ===============================
def split_text_into_chunks(text: str, max_chars: int = 3500) -> list[str]:
    """
    Аккуратный разбор длинной транскрибации на чанки,
    подходящие под контекст окна модели.

    - Нормализует пробелы
    - Делит по окончаниям предложений (.?!)
    - Собирает предложения в чанки по ограничению символов
    """
    # Нормализация пробелов и переводов строк
    normalized = re.sub(r"\s+", " ", text.strip())

    # Делим по концам предложений, но сохраняем знак препинания
    sentences = re.split(r"(?<=[\.\?\!])\s+", normalized)

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sent in sentences:
        if not sent:
            continue

        sent_len = len(sent) + 1  # +1 за пробел/разделитель
        if current and current_len + sent_len > max_chars:
            chunks.append(" ".join(current).strip())
            current = [sent]
            current_len = sent_len
        else:
            current.append(sent)
            current_len += sent_len

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def _chunk_list_by_length(items: list[str], max_chars: int) -> list[list[str]]:
    """
    Хелпер для иерархической компрессии:
    группирует строки в пачки так, чтобы суммарная длина
    каждой пачки не превышала max_chars.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    current_len = 0

    for item in items:
        if not item:
            continue

        item_len = len(item) + 1  # +1 за разделитель между блоками
        if current and current_len + item_len > max_chars:
            batches.append(current)
            current = [item]
            current_len = item_len
        else:
            current.append(item)
            current_len += item_len

    if current:
        batches.append(current)

    return batches


# ===============================
# СУММАРИЗАЦИЯ (CPU ONLY)
# ===============================
def generate_summary(transcription_text: str) -> str:
    n_ctx = 4096

    # Чуть увеличиваем лимиты генерации,
    # чтобы модель могла удержать больше тем и деталей.
    max_tokens_chunk = 384
    max_tokens_compress = 768
    max_tokens_final = 1024

    print("[summary] Loading LLM model (CPU only)...", flush=True)

    llm = Llama.from_pretrained(
        repo_id="ruslandev/llama-3-8b-gpt-4o-ru1.0-gguf",
        filename="ggml-model-Q2_K.gguf",
        n_ctx=n_ctx,
        n_threads=os.cpu_count(),
    )

    print("[summary] Model loaded.", flush=True)

    # 1. Разбиваем длинную транскрибацию на текстовые чанки
    chunks = split_text_into_chunks(transcription_text)
    print(f"[summary] Разбито на {len(chunks)} фрагментов", flush=True)

    if not chunks:
        return "Пустая транскрибация."

    partial_summaries: list[str] = []

    # ===============================
    # ЭТАП 1 — ТОЧНАЯ ФИКСАЦИЯ ФАКТОВ ПО ТЕМАМ
    # ===============================
    pbar = tqdm(chunks, desc="Summarizing", unit="chunk", file=sys.stdout)

    for chunk in pbar:
        prompt = f"""
Зафиксируй ФАКТЫ и ОТДЕЛЬНЫЕ ТЕМЫ из этого фрагмента встречи.

СТРОГИЕ ПРАВИЛА:
- НЕ обобщай
- НЕ используй слова: "обсуждали", "рассматривали", "поговорили"
- Пиши максимально близко к тексту
- Если нет решений — напиши: "Решений не зафиксировано"
- НИЧЕГО не добавляй от себя

НУЖНО ВЫДЕЛИТЬ:
- для КАЖДОЙ ОТДЕЛЬНОЙ ТЕМЫ/ВОПРОСА:
  - краткое название темы
  - конкретные высказывания
  - пояснения
  - варианты решений (если были)
  - принятые решения
  - договорённости и следующие шаги
  - замечания и уточнения

ФОРМАТ:
- Тема: <краткое название>
  - Факт: ...
  - Факт: ...
  - Решение: ...
  - Договорённость: ...
- Тема: <краткое название>
  - ...

ТЕКСТ:
{chunk}
""".strip()

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens_chunk,
            temperature=0.1,
            repeat_penalty=1.15,
        )

        partial_summaries.append(
            response["choices"][0]["message"]["content"].strip()
        )

    # ===============================
    # ЭТАП 2 — КОНСОЛИДАЦИЯ БЕЗ ВОДЫ (ИЕРАРХИЧЕСКАЯ, С СОХРАНЕНИЕМ ВСЕХ ТЕМ)
    # ===============================
    print("[summary] Compressing summaries...", flush=True)

    # Сначала сжимаем чанки частичных саммари партиями,
    # чтобы не переполнить контекст модели на длинных встречах.
    # 6000–7000 символов текста записей + промпт ≈ безопасно для n_ctx=4096.
    first_level_batches = _chunk_list_by_length(partial_summaries, max_chars=6500)
    first_level_protocols: list[str] = []

    for batch in tqdm(
        first_level_batches,
        desc="Compressing level-1",
        unit="batch",
        file=sys.stdout,
    ):
        compress_prompt = f"""
На основе записей ниже составь СТРОГИЙ ПРОТОКОЛ ВСТРЕЧИ,
СГРУППИРОВАННЫЙ ПО ТЕМАМ/ВОПРОСАМ.

ЖЁСТКИЕ ТРЕБОВАНИЯ:
- Используй ТОЛЬКО информацию из текста
- НЕ придумывай
- НЕ используй общие формулировки
- Объедини повторы ВНУТРИ одной и той же темы,
  но НЕ смешивай разные темы в одну
- Сохрани ВСЕ обсуждаемые темы, даже второстепенные
- Если блок пуст — НЕ ВЫВОДИ его

СТРУКТУРА:
ТЕМА: <краткое название темы>
- Вопрос: ...
- Ключевые аргументы и факты: ...
- Принятое решение: ... (или "Решение не принято")
- Договорённости и следующие шаги: ...

ТЕМА: <краткое название темы>
- Вопрос: ...
- Ключевые аргументы и факты: ...
- Принятое решение: ...
- Договорённости и следующие шаги: ...

ЗАПИСИ:
{chr(10).join(batch)}
""".strip()

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": compress_prompt}],
            max_tokens=max_tokens_compress,
            temperature=0.1,
            repeat_penalty=1.15,
        )

        first_level_protocols.append(
            response["choices"][0]["message"]["content"].strip()
        )

    # Если батч всего один — уже есть финальный протокол.
    if len(first_level_protocols) == 1:
        compressed = first_level_protocols[0]
    else:
        # Иначе делаем ещё один проход консолидации
        print("[summary] Consolidating compressed summaries...", flush=True)
        second_level_prompt = f"""
На основе протоколов ниже составь ЕДИНЫЙ СВОДНЫЙ ПРОТОКОЛ ВСТРЕЧИ,
СОХРАНЯЮЩИЙ ВСЕ ТЕМЫ/ВОПРОСЫ.

ЖЁСТКИЕ ТРЕБОВАНИЯ:
- Используй ТОЛЬКО информацию из текста
- НЕ придумывай
- Объедини повторы и противоречия ВНУТРИ одной темы,
  но НЕ сливай разные темы в одну
- Сохрани все темы, даже если они кажутся менее важными
- Если блок пуст — НЕ ВЫВОДИ его

СТРУКТУРА:
ТЕМА: <краткое название темы>
- Вопрос: ...
- Ключевые аргументы и факты: ...
- Принятое решение: ... (или "Решение не принято")
- Договорённости и следующие шаги: ...

ПРОТОКОЛЫ:
{chr(10).join(first_level_protocols)}
""".strip()

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": second_level_prompt}],
            max_tokens=max_tokens_compress,
            temperature=0.1,
            repeat_penalty=1.15,
        )

        compressed = response["choices"][0]["message"]["content"].strip()

    # ===============================
    # ЭТАП 3 — РЕЗЮМЕ КАК ОТ АНАЛИТИКА
    # ===============================
    print("[summary] Generating final summary...", flush=True)

    final_prompt = f"""
Ты опытный бизнес-аналитик и секретарь встречи.

На основе ПРОТОКОЛА ниже составь РЕЗЮМЕ ВСТРЕЧИ.

СТРОГИЕ ПРАВИЛА:
- НЕ выдумывай
- НЕ добавляй интерпретаций
- Пиши конкретно и деловым языком
- Резюме должно быть понятно руководству
- Если информации для блока нет — НЕ ВЫВОДИ его

ФОРМАТ РЕЗЮМЕ:

ТЕМА ВСТРЕЧИ:
(сформулируй по фактическому содержанию)

ОБСУЖДАЕМЫЕ ВОПРОСЫ:
- вопрос → суть

ПРИНЯТЫЕ РЕШЕНИЯ:
- решение → по какому вопросу

ДОГОВОРЁННОСТИ И СЛЕДУЮЩИЕ ШАГИ:
- кто → что должен сделать → при каких условиях

ПРОТОКОЛ:
{compressed}
""".strip()

    final_response = llm.create_chat_completion(
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=max_tokens_final,
        temperature=0.1,
        repeat_penalty=1.15,
    )

    return final_response["choices"][0]["message"]["content"].strip()


# ===============================
# СОХРАНЕНИЕ
# ===============================
def save_summary(output_path: str, summary: str):
    Path(output_path).write_text(summary, encoding="utf-8")
