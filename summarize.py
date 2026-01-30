"""
Структурированное резюме встречи: 3 этапа.
Этап 1: тематическая сегментация (маркеры, паузы, лексика).
Этап 2: анализ тем — решения, действия, ответственные, сроки.
Этап 3: структурированный вывод в формате ТЕМА / Обсуждалось / Результат / Действия.
"""

import json
import re
import torch
from pathlib import Path
from collections import Counter
from typing import Any

from transformers import GPT2Tokenizer, T5ForConditionalGeneration
from tqdm.auto import tqdm


# ===============================
# КОНФИГУРАЦИЯ
# ===============================
MODEL_NAME = "RussianNLP/FRED-T5-Summarizer"
DEVICE = "cpu"
MAX_CONTEXT_LENGTH = 1024
MAX_NEW_TOKENS_THEME = 220
MIN_NEW_TOKENS_THEME = 80
MAX_NEW_TOKENS_ACTIONS = 120
MIN_NEW_TOKENS_ACTIONS = 10
MIN_THEME_WORDS = 250
MAX_THEME_WORDS = 3000
MAX_THEMES_OUTPUT = 8
PAUSE_THRESHOLD_SEC = 3.0
WINDOW_SENTENCES = 5
KEYWORD_SIMILARITY_THRESHOLD = 0.4
TOP_KEYWORDS_PER_WINDOW = 12

# Маркеры смены тем (регулярки, регистр не важен)
MARKERS = [
    r"переходим\s+к",
    r"следующий\s+вопрос",
    r"теперь\s+о",
    r"по\s+следующему",
    r"по\s+поводу",
    r"далее\s+—",
    r"далее\s+по",
    r"следующая\s+тема",
    r"новый\s+вопрос",
    r"другой\s+вопрос",
    r"отдельно\s+по",
    r"вернёмся\s+к",
    r"перейдём\s+к",
    r"перейдем\s+к",
]
MARKER_RE = re.compile("|".join(f"({m})" for m in MARKERS), re.I)

# Стоп-слова для ключевых слов (краткий список)
STOPWORDS_RU = {
    "и", "в", "во", "не", "что", "на", "я", "с", "со", "как", "а", "то", "все",
    "она", "так", "он", "к", "но", "они", "мы", "вы", "её", "оно", "от", "у",
    "уже", "ни", "под", "о", "при", "до", "из", "когда", "за", "бы", "по",
    "для", "это", "тем", "чем", "или", "без", "его", "ей", "им", "их", "ли",
    "нет", "нам", "она", "от", "при", "себя", "те", "то", "тоже", "что",
    "чтобы", "эта", "эти", "этот", "этого", "этому", "нет", "да", "вот",
}


# ===============================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ===============================
def count_tokens(text: str, tokenizer) -> int:
    return len(tokenizer.encode(text, add_special_tokens=True))


def truncate_to_max_tokens(text: str, tokenizer, max_tokens: int) -> str:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens], skip_special_tokens=False)


# ===============================
# ЗАГРУЗКА ТРАНСКРИПЦИИ
# ===============================
def get_transcription_text(file_path: str) -> str:
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


def get_transcription_segments(file_path: str) -> list[dict] | None:
    """Сегменты с временными метками (только для JSON). Для TXT — None."""
    if not file_path.endswith(".json"):
        return None
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    return [
        {"text": s["text"].strip(), "start": float(s.get("start", 0)), "end": float(s.get("end", 0))}
        for s in data if s.get("text", "").strip()
    ]


def normalize_transcription(text: str) -> str:
    """Удаление [PAUSE], артефактов, нормализация пробелов."""
    t = re.sub(r"\[PAUSE\]", " ", text, flags=re.I)
    t = re.sub(r"\[[\w\s]+\]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ===============================
# ЭТАП 1: ТЕМАТИЧЕСКАЯ СЕГМЕНТАЦИЯ
# ===============================
def _sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.strip())
    s = re.split(r"(?<=[.!?…])\s+", normalized)
    return [x.strip() for x in s if x.strip()]


def _word_count(sentences: list[str]) -> int:
    return sum(len(re.findall(r"\w+", sent)) for sent in sentences)


def _keyword_set(sentences: list[str], top_n: int = TOP_KEYWORDS_PER_WINDOW) -> set[str]:
    words = []
    for s in sentences:
        for w in re.findall(r"[а-яёa-z0-9]+", s.lower()):
            if len(w) > 1 and w not in STOPWORDS_RU:
                words.append(w)
    cnt = Counter(words)
    return {x for x, _ in cnt.most_common(top_n)}


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 1.0


def _char_offsets(sentences: list[str]) -> list[int]:
    offsets = [0]
    for s in sentences:
        offsets.append(offsets[-1] + len(s) + 1)
    return offsets[:-1]


def detect_topics(text: str, segments: list[dict] | None = None) -> list[str]:
    """
    Разбивает текст на тематические блоки по маркерам, паузам и лексике.
    Возвращает список строк (текст каждой темы). При segments передают text,
    собранный из тех же сегментов (для согласованного маппинга пауз).
    """
    text = normalize_transcription(text)
    sentences = _sentences(text)
    if not sentences:
        return []

    n = len(sentences)
    boundaries = set()  # индексы предложений, перед которыми граница темы

    # 1) Маркеры
    full = " ".join(sentences)
    offsets = _char_offsets(sentences)
    for m in MARKER_RE.finditer(full):
        pos = m.start()
        for i, off in enumerate(offsets):
            if off >= pos:
                boundaries.add(i)
                break
        else:
            boundaries.add(n)

    # 2) Паузы > 3 сек (если есть сегменты); cumlen по сегментам в том же порядке, что text
    if segments:
        parts = [normalize_transcription(s["text"]) for s in segments]
        cumlen = [0]
        for p in parts:
            cumlen.append(cumlen[-1] + len(p) + 1)
        for i in range(1, len(segments)):
            gap = segments[i]["start"] - segments[i - 1]["end"]
            if gap >= PAUSE_THRESHOLD_SEC:
                pos = cumlen[i]
                for j, off in enumerate(offsets):
                    if off >= pos:
                        boundaries.add(j)
                        break
                else:
                    boundaries.add(n)

    # 3) Лексическое окно: схожесть < 0.4 → граница
    for i in range(WINDOW_SENTENCES, n):
        w1 = sentences[i - WINDOW_SENTENCES : i]
        w2 = sentences[i - WINDOW_SENTENCES + 1 : i + 1]
        if len(w2) < WINDOW_SENTENCES:
            continue
        k1 = _keyword_set(w1)
        k2 = _keyword_set(w2)
        if _jaccard(k1, k2) < KEYWORD_SIMILARITY_THRESHOLD:
            boundaries.add(i)

    # Собираем границы, сортируем
    bounds = sorted(set(boundaries) | {0, n})
    if bounds[0] != 0:
        bounds.insert(0, 0)
    if bounds[-1] != n:
        bounds.append(n)

    # Группы предложений между границами
    groups: list[list[str]] = []
    for j in range(len(bounds) - 1):
        group = sentences[bounds[j] : bounds[j + 1]]
        if group:
            groups.append(group)

    # Объединяем слишком короткие, разбиваем слишком длинные
    merged: list[list[str]] = []
    i = 0
    while i < len(groups):
        g = groups[i]
        wc = _word_count(g)
        if wc < MIN_THEME_WORDS and merged:
            merged[-1].extend(g)
        elif wc > MAX_THEME_WORDS:
            mid = len(g) // 2
            merged.append(g[:mid])
            merged.append(g[mid:])
        else:
            merged.append(g)
        i += 1

    # Финальное объединение коротких
    final: list[list[str]] = []
    for g in merged:
        if _word_count(g) < MIN_THEME_WORDS and final:
            final[-1].extend(g)
        else:
            final.append(g)

    # Разбить слишком длинные после слияния
    expanded: list[list[str]] = []
    for g in final:
        while _word_count(g) > MAX_THEME_WORDS:
            mid = len(g) // 2
            expanded.append(g[:mid])
            g = g[mid:]
        if g:
            expanded.append(g)
    final = expanded

    # Не больше MAX_THEMES_OUTPUT: объединяем самые короткие
    while len(final) > MAX_THEMES_OUTPUT:
        idx = min(range(len(final)), key=lambda i: _word_count(final[i]))
        if idx == 0:
            final[0] = final[0] + final[1]
            final.pop(1)
        else:
            final[idx - 1] = final[idx - 1] + final[idx]
            final.pop(idx)

    return [" ".join(g) for g in final if g]


# ===============================
# ЭТАП 2: АНАЛИЗ ТЕМЫ
# ===============================
def _extract_phrases(text: str, patterns: list[str]) -> list[str]:
    found = []
    for p in patterns:
        for m in re.finditer(p, text, re.I):
            fragment = m.group(1).strip() if m.lastindex and m.lastindex >= 1 else m.group(0)
            fragment = re.sub(r"\s+", " ", fragment)
            if len(fragment) > 5 and fragment not in found:
                found.append(fragment)
    return found


def analyze_topic(topic_text: str) -> dict[str, Any]:
    """
    Извлекает решения, действия (с ответственным и сроком), ключевые слова для названия.
    """
    t = normalize_transcription(topic_text)
    decisions = []
    actions = []

    # Решения
    decision_pats = [
        r"(?:решили|постановили|утвердили|договорились|согласовали|согласовано)\s*[:\—,]?\s*([^.!?]+[.!?])",
        r"(?:будем\s+делать|принимаем)\s*[:\—,]?\s*([^.!?]+[.!?])",
        r"в\s+итоге\s+[^.]*?([^.!?]+[.!?])",
    ]
    decisions = _extract_phrases(t, decision_pats)

    # Действия: "подготовить ... до пятницы", "Ивану подготовить ..."
    action_pats = [
        r"(?:нужно|надо|требуется|необходимо)\s+(?:сделать|подготовить|предоставить|согласовать|провести|устранить|составить|проверить)[^.!?]*[.!?]",
        r"(?:поручается|назначить|назначен)\s+[^.!?]*[.!?]",
        r"(?:подготовить|предоставить|согласовать|провести|устранить|составить|проверить)\s+[^.!?]*(?:до|к|на|в\s+течение|через|через месяц)[^.!?]*[.!?]",
        r"[А-Яа-яёЁ]+\s+(?:подготовить|предоставить|согласовать|провести|устранить|составить|проверить)[^.!?]*[.!?]",
    ]
    for p in action_pats:
        for m in re.finditer(p, t, re.I):
            frag = m.group(0).strip()
            frag = re.sub(r"\s+", " ", frag)
            if len(frag) > 10 and frag not in actions:
                actions.append(frag)

    # Название темы — 2–4 ключевых слова
    words = []
    for w in re.findall(r"[а-яё0-9]+", t.lower()):
        if len(w) > 2 and w not in STOPWORDS_RU:
            words.append(w)
    cnt = Counter(words)
    title_keywords = [x for x, _ in cnt.most_common(4)]

    return {
        "decisions": decisions,
        "actions": actions,
        "title_keywords": title_keywords,
    }


def theme_title_from_keywords(keywords: list[str]) -> str:
    if not keywords:
        return "Обсуждение"
    return " ".join(keywords[:4]).capitalize()


# ===============================
# ЭТАП 3: СУММАРИЗАЦИЯ ТЕМЫ И ВЫВОД
# ===============================
def summarize_theme_t5(topic_text: str, model, tokenizer, device: str) -> tuple[str, str]:
    """
    Возвращает (обсуждалось, результат) для темы. Короткий промпт, только факты.
    """
    prefix = (
        "<LM> Составь подробное резюме фрагмента встречи. "
        "Сначала опиши, что именно обсуждалось (2–4 предложения, с перечислением ключевых аргументов, фактов, цифр и вариантов). "
        "Затем опиши итог/решение (1–3 предложения, с акцентом на конкретные договорённости). "
        "Не придумывай, используй только факты из текста, без имён участников. Текст:\n "
    )
    max_input = MAX_CONTEXT_LENGTH - count_tokens(prefix, tokenizer) - MAX_NEW_TOKENS_THEME - 50
    raw = truncate_to_max_tokens(topic_text, tokenizer, max(100, max_input))
    full = prefix + raw
    input_ids = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS_THEME,
    ).input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS_THEME,
            min_new_tokens=MIN_NEW_TOKENS_THEME,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    summary = tokenizer.decode(out[0][1:], skip_special_tokens=True).strip()

    # Пытаемся разделить на "обсуждалось" и "результат" по точке или "итог"
    obs, res = "", ""
    if ". " in summary:
        parts = summary.split(". ", 1)
        obs = parts[0].strip()
        if len(parts) > 1:
            res = parts[1].strip()
    else:
        obs = summary
    if not res and ("итог" in summary.lower() or "решили" in summary.lower() or "договорились" in summary.lower()):
        idx = max(
            summary.lower().find("итог"),
            summary.lower().find("решили"),
            summary.lower().find("договорились"),
        )
        if idx >= 0:
            obs = summary[:idx].strip()
            res = summary[idx:].strip()
    if not obs:
        obs = summary
    if not res:
        res = "Обмен мнениями, решение не зафиксировано."
    return obs, res


def extract_actions_t5(topic_text: str, model, tokenizer, device: str) -> list[str]:
    """
    Извлекает действия/поручения с ответственными в формате:
    Имя - что должен сделать (кратко).
    """
    prefix = (
        "<LM> Извлеки из текста конкретные задачи и поручения. "
        "Верни только список строк в формате: Имя - что должен сделать (кратко). "
        "Если ответственный явно не указан, используй 'Не назначено' как имя. "
        "Не добавляй пояснений, вводных фраз и лишнего текста, только задачи, по одной на строке. Текст:\n "
    )
    max_input = MAX_CONTEXT_LENGTH - count_tokens(prefix, tokenizer) - MAX_NEW_TOKENS_ACTIONS - 50
    raw = truncate_to_max_tokens(topic_text, tokenizer, max(100, max_input))
    full = prefix + raw
    input_ids = tokenizer(
        full,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_CONTEXT_LENGTH - MAX_NEW_TOKENS_ACTIONS,
    ).input_ids.to(device)

    with torch.no_grad():
        out = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS_ACTIONS,
            min_new_tokens=MIN_NEW_TOKENS_ACTIONS,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    text = tokenizer.decode(out[0][1:], skip_special_tokens=True).strip()
    lines = [re.sub(r"\s+", " ", ln).strip(" •-*—\t") for ln in text.splitlines()]
    lines = [ln for ln in lines if len(ln) > 5]
    return _deduplicate_lines(lines)


def _deduplicate_lines(lines: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in lines:
        x = re.sub(r"\s+", " ", x).strip()
        key = x.lower()[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(x)
    return out


def format_structured_summary(themes: list[dict]) -> str:
    """
    Генерирует финальный текст в формате:
    ТЕМА N: ...
    - Обсуждалось: ...
    - Результат: ...
    - Действия: ...
    """
    blocks = []
    for i, th in enumerate(themes, 1):
        obs = th.get("obsuzdalos") or "Обсуждение по теме."
        res = th.get("result")
        if not res:
            res = "Обмен мнениями, решение отложено." if not th.get("decisions") else " ".join(th["decisions"][:2])
        actions = th.get("actions") or []
        actions = _deduplicate_lines(actions)
        if not actions:
            actions = ["Нет назначенных действий."]

        block = [
            f"ТЕМА {i}",
            f"- Обсуждалось: {obs}",
            f"- Результат: {res}",
            "- Действия:",
        ]
        for a in actions[:5]:
            block.append(f"  • {a}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


# ===============================
# ОСНОВНАЯ ФУНКЦИЯ СУММАРИЗАЦИИ
# ===============================
def generate_summary(transcription_text: str, file_path: str | None = None) -> str:
    """
    Пайплайн: тематическая сегментация → анализ тем → T5 (обсуждалось/результат) → структурированный вывод.
    file_path нужен для пауз (JSON с сегментами). Можно не передавать.
    """
    print("[summary] Загрузка модели FRED-T5...")
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME, eos_token="</s>")
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    segments = get_transcription_segments(file_path) if file_path else None
    if segments:
        text = normalize_transcription(" ".join(s["text"] for s in segments))
    else:
        text = normalize_transcription(transcription_text)
    if not text.strip():
        return "Ошибка: пустая транскрипция."

    # Этап 1: тематическая сегментация
    print("[summary] Этап 1: Тематическая сегментация...")
    topics = detect_topics(text, segments)
    print(f"[summary] Выделено тем: {len(topics)}")

    if not topics:
        return "Не удалось выделить темы. Проверьте формат транскрипции."

    # Этап 2: анализ каждой темы + T5 для «обсуждалось» / «результат»
    themes_data = []
    for topic_text in tqdm(topics, desc="Темы", unit="шт"):
        ana = analyze_topic(topic_text)
        obs, res = summarize_theme_t5(topic_text, model, tokenizer, DEVICE)
        # Действия формируем через T5 в формате [Имя] - [что сделать],
        # при отсутствии — используем эвристическое извлечение как запасной вариант.
        actions_llm = extract_actions_t5(topic_text, model, tokenizer, DEVICE)
        actions = actions_llm if actions_llm else ana["actions"]
        themes_data.append({
            "title_keywords": ana["title_keywords"],
            "decisions": ana["decisions"],
            "actions": actions,
            "obsuzdalos": obs,
            "result": res,
        })

    # Этап 3: структурированный вывод
    print("[summary] Этап 3: Формирование структурированного резюме...")
    out = format_structured_summary(themes_data)

    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return out


def save_summary(output_path: str, summary: str) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(summary, encoding="utf-8")
    print(f"[summary] Резюме сохранено: {output_path}")
