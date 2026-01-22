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

def split_text_into_chunks(text: str, max_chars: int = 6000) -> list[str]:
    sentences = text.split(". ")
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

def generate_summary_for_chunk(chunk: str, llm) -> str:
    prompt = f"""Извлеки из фрагмента пары: "вопрос — итог".
Если итога нет — напиши "без решения".
Фрагмент: {chunk}
Результат:"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=256,
        temperature=0.0,
        repeat_penalty=1.1
    )
    return output["choices"][0]["message"]["content"].strip()

def generate_final_summary(summaries: list[str], llm) -> str:
    # Фильтрация пустых и бессмысленных чанков
    filtered_summaries = [
        s for s in summaries 
        if s and "нет" not in s.lower() and "пропусти" not in s.lower() and len(s) > 10
    ]
    
    if not filtered_summaries:
        return "Не удалось извлечь значимые обсуждения из транскрибации."
    
    combined = "\n".join(filtered_summaries)
    
    final_prompt = f"""Создай резюме встречи по шаблону. Используй ТОЛЬКО факты из текста. Не добавляй ничего от себя.

ТЕМА ВСТРЕЧИ: [одна фраза]
ОБСУЖДАЕМЫЕ ВОПРОСЫ:
- [вопрос]: [итог]

Текст:
{combined}

Резюме:"""

    messages = [
        {"role": "user", "content": final_prompt}
    ]
    
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=768,
        temperature=0.0,
        repeat_penalty=1.1
    )
    return output["choices"][0]["message"]["content"].strip()

def generate_summary(transcription_text: str, model_path: str) -> str:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8,
        n_gpu_layers=0,
        verbose=False
    )
    
    chunks = split_text_into_chunks(transcription_text, max_chars=6000)
    print(f"Разбито на {len(chunks)} фрагментов")
    
    chunk_summaries = []
    for i, chunk in enumerate(chunks):
        print(f"Обработка фрагмента {i+1}/{len(chunks)}...")
        summary = generate_summary_for_chunk(chunk, llm)
        chunk_summaries.append(summary)
    
    print("Генерация итогового резюме по шаблону...")
    final_summary = generate_final_summary(chunk_summaries, llm)
    return final_summary

def save_summary(output_path: str, summary: str):
    Path(output_path).write_text(summary, encoding="utf-8")