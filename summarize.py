import os
import json
import sys
from pathlib import Path
from llama_cpp import Llama
from tqdm.auto import tqdm


def get_transcription_text(file_path: str) -> str:
    if file_path.endswith(".json"):
        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        return "\n".join(seg["text"] for seg in data if seg.get("text", "").strip())
    elif file_path.endswith(".txt"):
        lines = Path(file_path).read_text(encoding="utf-8").splitlines()
        texts = []
        for line in lines:
            if "] " in line:
                parts = line.split("] ", 1)
                if len(parts) == 2:
                    texts.append(parts[1])
                else:
                    texts.append(line)
            else:
                texts.append(line)
        return "\n".join(texts)
    else:
        raise ValueError("Only .txt and .json supported")

def split_text_into_chunks(text: str, max_chars: int = 4500) -> list[str]:
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current_chunk = ""
    for sent in sentences:
        if len(current_chunk) + len(sent) + 2 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sent + ". "
        else:
            current_chunk += sent + ". "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def generate_summary(transcription_text: str) -> str:
    # "Больше токенов" бывает двух типов:
    # - n_ctx: размер контекстного окна модели (сколько токенов влезает во вход+выход)
    # - max_tokens: сколько токенов генерировать в ответ
    n_ctx = int(os.getenv("LLM_N_CTX", "4096"))
    max_tokens_chunk = int(os.getenv("LLM_MAX_TOKENS_CHUNK", "1024"))
    max_tokens_final = int(os.getenv("LLM_MAX_TOKENS_FINAL", "1024"))

    print("[summary] Loading LLM model...", flush=True)
    llm = Llama.from_pretrained(
        repo_id="IlyaGusev/saiga_llama3_8b_gguf",
	    filename="model-q8_0.gguf",
        n_ctx=n_ctx,
    )
    print("[summary] Model loaded.", flush=True)
    chunks = split_text_into_chunks(transcription_text, max_chars=4500)
    print(f"[summary] Разбито на {len(chunks)} фрагментов", flush=True)
    
    if not chunks:
        return "Пустая транскрибация."
    
    # Начинаем с пустого списка
    current_summary = ""
    
    pbar = tqdm(
        chunks,
        desc="Summarizing",
        unit="chunk",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=0.5,
    )
    for i, chunk in enumerate(pbar):
        tqdm.write(f"[summary] Обработка фрагмента {i+1}/{len(chunks)}...")
        
        # IMPORTANT: build a single string (your previous version only kept the first line)
        prompt = (
            "Суммаризируй текст беседы, сохраняя основные положения из предыдущего резюме.\n"
            "Выдели основные темы/вопросы и их решения.\n"
            f"===Предыдущее резюме===\n{current_summary}\n"
            f"===Текущий текст беседы===\n{chunk}"
        )
        
        messages = [{"role": "user", "content": prompt}]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens_chunk,
            temperature=0.3,
            repeat_penalty=1.1
        )
        current_summary = response["choices"][0]["message"]["content"].strip()
    
    # Финальное оформление
    final_prompt = f"""Ты секретарь, который ведет протокол встречи. Преобразуй протокол в официальное резюме встречи.

===Протокол===
{current_summary}

===Формат===
ТЕМА ВСТРЕЧИ: [общая тема на основе вопросов]
ОБСУЖДАЕМЫЕ ВОПРОСЫ:
[список из протокола]

ИТОГОВОЕ РЕЗЮМЕ: [резюме]"""
    # printing the entire final prompt can be huge; keep logs compact
    print("[summary] Generating final formatted summary...", flush=True)
    messages = [{"role": "user", "content": final_prompt}]
    final_response = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens_final,
        temperature=0.3,
        repeat_penalty=1.1
    )
    return final_response["choices"][0]["message"]["content"].strip()

def save_summary(output_path: str, summary: str):
    Path(output_path).write_text(summary, encoding="utf-8")