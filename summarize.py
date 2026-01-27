import os
import json
import sys
from pathlib import Path
from llama_cpp import Llama
from tqdm.auto import tqdm


# ===============================
# ЗАГРУЗКА ТРАНСКРИПЦИИ
# ===============================
def get_transcription_text(file_path: str) -> str:
    if file_path.endswith(".json"):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return "\n".join(seg["text"] for seg in data if seg.get("text", "").strip())

    elif file_path.endswith(".txt"):
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        texts = []
        for line in lines:
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
def split_text_into_chunks(text: str, max_chars: int = 4200) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) > max_chars:
            chunks.append(current.strip())
            current = sent + ". "
        else:
            current += sent + ". "

    if current.strip():
        chunks.append(current.strip())

    return chunks


# ===============================
# СУММАРИЗАЦИЯ (CPU ONLY)
# ===============================
def generate_summary(transcription_text: str) -> str:
    n_ctx = 4096

    max_tokens_chunk = 256
    max_tokens_compress = 256
    max_tokens_final = 512

    print("[summary] Loading LLM model (CPU only)...", flush=True)

    llm = Llama.from_pretrained(
        repo_id="ruslandev/llama-3-8b-gpt-4o-ru1.0-gguf",
	    filename="ggml-model-Q2_K.gguf",   # ⚠️ КЛЮЧЕВО
        n_ctx=n_ctx,
        n_threads=os.cpu_count(),       # использовать все ядра
    )

    print("[summary] Model loaded.", flush=True)

    chunks = split_text_into_chunks(transcription_text)
    print(f"[summary] Разбито на {len(chunks)} фрагментов", flush=True)

    if not chunks:
        return "Пустая транскрибация."

    partial_summaries = []

    # ===============================
    # ЭТАП 1 — МИКРО-РЕЗЮМЕ
    # ===============================
    pbar = tqdm(chunks, desc="Summarizing", unit="chunk", file=sys.stdout)

    for chunk in pbar:
        prompt = f"""
Кратко (5–7 пунктов) опиши содержание беседы.

Правила:
- только факты
- темы, решения, договорённости
- без воды

Текст:
{chunk}
""".strip()

        response = llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens_chunk,
            temperature=0.2,
            repeat_penalty=1.1,
        )

        partial_summaries.append(
            response["choices"][0]["message"]["content"].strip()
        )

    # ===============================
    # ЭТАП 2 — СЖАТИЕ
    # ===============================
    print("[summary] Compressing summaries...", flush=True)

    compress_prompt = f"""
Сожми информацию ниже в краткий протокол встречи.

Оставь:
- ключевые темы
- принятые решения
- договорённости

Информация:
{chr(10).join(partial_summaries)}
""".strip()

    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": compress_prompt}],
        max_tokens=max_tokens_compress,
        temperature=0.2,
        repeat_penalty=1.1,
    )

    compressed = response["choices"][0]["message"]["content"].strip()

    # ===============================
    # ЭТАП 3 — ФИНАЛ
    # ===============================
    print("[summary] Generating final summary...", flush=True)

    final_prompt = f"""
Ты секретарь, ведущий протокол встречи.
Сформируй официальное резюме.

=== ПРОТОКОЛ ===
{compressed}

=== ФОРМАТ ===
ТЕМА ВСТРЕЧИ:
ОБСУЖДАЕМЫЕ ВОПРОСЫ:
ИТОГОВЫЕ РЕШЕНИЯ:
ДОГОВОРЁННОСТИ:
""".strip()

    final_response = llm.create_chat_completion(
        messages=[{"role": "user", "content": final_prompt}],
        max_tokens=max_tokens_final,
        temperature=0.2,
        repeat_penalty=1.1,
    )

    return final_response["choices"][0]["message"]["content"].strip()


# ===============================
# СОХРАНЕНИЕ
# ===============================
def save_summary(output_path: str, summary: str):
    Path(output_path).write_text(summary, encoding="utf-8")
