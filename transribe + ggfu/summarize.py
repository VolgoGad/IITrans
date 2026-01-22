import os
import json
from pathlib import Path
from llama_cpp import Llama

os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

def generate_summary(transcription_text: str, model_path: str) -> str:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=12,
        n_gpu_layers=0,
        verbose=False
    )
    
    chunks = split_text_into_chunks(transcription_text, max_chars=4500)
    print(f"Разбито на {len(chunks)} фрагментов")
    
    if not chunks:
        return "Пустая транскрибация."
    
    # Начинаем с пустого списка
    current_summary = "Нет обсуждений."
    
    for i, chunk in enumerate(chunks):
        print(f"Обработка фрагмента {i+1}/{len(chunks)}...")
        
        prompt = f"""Ты ведёшь протокол встречи. У тебя есть текущий протокол и новый фрагмент транскрипции.
Добавь из фрагмента ТОЛЬКО новые пары "вопрос — итог".
Если итога нет — пиши "без решения".
Не повторяй уже существующие пункты.

ФОРМАТ ВЫВОДА:
- [вопрос]: [итог]

ТЕКУЩИЙ ПРОТОКОЛ:
{current_summary}

НОВЫЙ ФРАГМЕНТ:
{chunk}

ОБНОВЛЁННЫЙ ПРОТОКОЛ:"""
        
        messages = [{"role": "user", "content": prompt}]
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.0,
            repeat_penalty=1.1
        )
        current_summary = response["choices"][0]["message"]["content"].strip()
    
    # Финальное оформление
    final_prompt = f"""Преобразуй протокол в официальное резюме встречи.

Протокол:
{current_summary}

Формат:
ТЕМА ВСТРЕЧИ: [общая тема на основе вопросов]
ОБСУЖДАЕМЫЕ ВОПРОСЫ:
[список из протокола]

ИТОГОВОЕ РЕЗЮМЕ:"""
    
    messages = [{"role": "user", "content": final_prompt}]
    final_response = llm.create_chat_completion(
        messages=messages,
        max_tokens=768,
        temperature=0.0,
        repeat_penalty=1.1
    )
    return final_response["choices"][0]["message"]["content"].strip()

def save_summary(output_path: str, summary: str):
    Path(output_path).write_text(summary, encoding="utf-8")