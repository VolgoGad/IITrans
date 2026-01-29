import os
import json
import re
import torch
from pathlib import Path
from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm


# ===============================
# КОНФИГУРАЦИЯ
# ===============================
MODEL_NAME = "RussianNLP/FRED-T5-Summarizer"
DEVICE = "cpu"  # "cuda" если есть GPU
MAX_CONTEXT_LENGTH = 1024  # лимит токенов для FRED-T5 (вход + выход)
MAX_NEW_TOKENS_CHUNK = 200  # длина вывода для чанков
MIN_NEW_TOKENS_CHUNK = 17
MAX_NEW_TOKENS_FINAL = 400   # длина финального резюме
MIN_NEW_TOKENS_FINAL = 100
CHUNK_SIZE_CHARS = 800       # ~600-700 токенов после токенизации + запас на префикс
OVERLAP_CHARS = 150


# ===============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================
def count_tokens(text: str, tokenizer) -> int:
    """Подсчёт токенов с учётом специальных токенов."""
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Обрезка текста до заданного количества токенов."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    truncated = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=False)
    return truncated


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
        return "\n".join(seg["text"].strip() for seg in data if seg.get("text", "").strip())

    elif file_path.endswith(".txt"):
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        texts = []
        for line in lines:
            if "] " in line:
                texts.append(line.split("] ", 1)[1].strip())
            elif line.strip():
                texts.append(line.strip())
        return "\n".join(texts)

    else:
        raise ValueError("Поддерживаются только .txt и .json файлы")


# ===============================
# ЧАНКИНГ С КОРРЕКТНЫМ ОВЕРЛАПОМ
# ===============================
def split_text_into_chunks(text: str, max_chars: int = CHUNK_SIZE_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """
    Разбивает текст на чанки с фиксированным оверлапом по символам.
    """
    # Нормализация пробелов
    normalized = re.sub(r"\s+", " ", text.strip())
    
    # Делим по предложениям с поддержкой русской пунктуации
    sentences = re.split(r'(?<=[.!?…])\s+', normalized)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        sentence_length = len(sentence) + 1  # +1 для пробела
        
        if current_chunk and current_length + sentence_length > max_chars:
            # Сохраняем текущий чанк
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text.strip())
            
            # Рассчитываем оверлап по символам
            if overlap > 0 and chunk_text:
                overlap_text = chunk_text[-overlap:].strip()
                # Находим последнюю границу предложения в оверлапе
                overlap_sentences = re.split(r'(?<=[.!?…])\s+', overlap_text)
                if len(overlap_sentences) > 1:
                    overlap_text = ' '.join(overlap_sentences[1:])  # пропускаем неполное предложение
                
                current_chunk = [overlap_text, sentence] if overlap_text else [sentence]
                current_length = len(overlap_text) + 1 + len(sentence)
            else:
                current_chunk = [sentence]
                current_length = len(sentence)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    # Добавляем последний чанк
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    return chunks


# ===============================
# СУММАРИЗАЦИЯ ЧАНКА
# ===============================
def summarize_chunk(chunk: str, model, tokenizer, device) -> str:
    """
    Суммаризует один чанк с правильным форматированием для FRED-T5.
    """
    # Обязательный префикс для FRED-T5
    input_text = f"<LM> Сократи текст.\n {chunk.strip()}"
    
    # Токенизация с обрезкой до лимита контекста
    input_ids = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS_CHUNK  # оставляем место для вывода
    ).input_ids.to(device)
    
    # Генерация
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS_CHUNK,
            min_new_tokens=MIN_NEW_TOKENS_CHUNK,
            num_beams=5,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=4,
            early_stopping=True
        )
    
    # Декодирование (пропускаем первый токен — обычно <pad>)
    summary = tokenizer.decode(outputs[0][1:], skip_special_tokens=True)
    return summary.strip()


# ===============================
# ФИНАЛЬНАЯ СУММАРИЗАЦИЯ
# ===============================
def create_final_summary(combined_text: str, model, tokenizer, device) -> str:
    """
    Создаёт структурированное финальное резюме.
    Важно: промпт должен умещаться в контекстное окно!
    """
    # Упрощённый промпт (длинные инструкции превысят лимит 1024 токена)
    prompt = f"<LM> Сделай краткое структурированное резюме обсуждения.\n {combined_text.strip()}"
    
    # Проверка и обрезка до безопасного размера
    if count_tokens(prompt, tokenizer) > MAX_CONTEXT_LENGTH - MIN_NEW_TOKENS_FINAL:
        max_input_tokens = MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS_FINAL - 50  # запас на префикс
        prompt = f"<LM> Сделай краткое структурированное резюме обсуждения.\n {truncate_to_max_tokens(combined_text, tokenizer, max_input_tokens)}"
        print(f"[!] Финальный текст обрезан до {max_input_tokens} токенов для умещения в контекст")
    
    input_ids = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS_FINAL
    ).input_ids.to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS_FINAL,
            min_new_tokens=MIN_NEW_TOKENS_FINAL,
            num_beams=5,
            do_sample=True,
            top_p=0.9,
            no_repeat_ngram_size=4,
            early_stopping=True
        )
    
    summary = tokenizer.decode(outputs[0][1:], skip_special_tokens=True)
    return summary.strip()


# ===============================
# ОСНОВНАЯ ФУНКЦИЯ СУММАРИЗАЦИИ
# ===============================
def generate_summary(transcription_text: str) -> str:
    """
    Полный пайплайн суммаризации с иерархическим подходом.
    """
    print("[summary] Загрузка модели FRED-T5...")
    
    # Загрузка правильного токенизатора (на базе GPT-2)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, eos_token='</s>')
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()  # режим инференса
    
    print(f"[summary] Модель загружена. Устройство: {DEVICE.upper()}")
    
    # 1. Разбиваем на чанки
    chunks = split_text_into_chunks(transcription_text, max_chars=CHUNK_SIZE_CHARS, overlap=OVERLAP_CHARS)
    print(f"[summary] Текст разбит на {len(chunks)} чанков")
    
    if not chunks:
        return "Ошибка: пустая транскрибация."
    
    # 2. Суммаризация каждого чанка
    print("[summary] Этап 1: Суммаризация чанков...")
    chunk_summaries = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Чанки", unit="шт")):
        try:
            summary = summarize_chunk(chunk, model, tokenizer, DEVICE)
            chunk_summaries.append(summary)
        except Exception as e:
            print(f"\nОшибка при обработке чанка {i}: {e}")
            # Fallback: короткий выдержка из чанка
            chunk_summaries.append(f"Краткое содержание: {chunk[:150]}...")
    
    # 3. Объединение промежуточных резюме
    print("[summary] Этап 2: Объединение промежуточных резюме...")
    combined_summaries = " ".join(chunk_summaries)
    
    # При необходимости — промежуточное сжатие
    if count_tokens(combined_summaries, tokenizer) > 700:
        print("[summary] Промежуточное сжатие объединённого текста...")
        combined_summaries = summarize_chunk(combined_summaries, model, tokenizer, DEVICE)
    
    # 4. Финальная суммаризация
    print("[summary] Этап 3: Финальное структурированное резюме...")
    final_summary = create_final_summary(combined_summaries, model, tokenizer, DEVICE)
    
    # Очистка памяти
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    
    return final_summary


# ===============================
# СОХРАНЕНИЕ РЕЗУЛЬТАТА
# ===============================
def save_summary(output_path: str, summary: str):
    """
    Сохраняет резюме в файл.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(summary, encoding="utf-8")
    print(f"[summary] Резюме сохранено: {output_path}")

