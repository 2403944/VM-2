"""Standalone text-to-chunks-to-questions utility.

Use this module when text is already available and only questions plus their
source context are required. It does not upload files, index vectors, persist
records, or generate answers.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any


LLMGenerator = Callable[[str], str]
PromptBuilder = Callable[..., str]


@dataclass(frozen=True)
class QuestionRecord:
    """A generated question and the chunk that supplied its context."""

    question: str
    context: str
    chunk_index: int

    def to_dict(self) -> dict[str, Any]:
        """Return the record in an API- and JSON-friendly shape."""
        return asdict(self)


def chunk_content(
    content: str,
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
    separators: Sequence[str] | None = None,
) -> list[str]:
    """Split text with LangChain's recursive character splitter.

    Args:
        content: Source text to split.
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Characters shared by adjacent chunks.
        separators: Optional separator priority for the recursive splitter.

    Returns:
        Non-empty text chunks in source order.

    Raises:
        ValueError: If the content or chunking values are invalid.
    """
    if not isinstance(content, str) or not content.strip():
        raise ValueError("content must be a non-empty string")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be zero or greater and smaller than chunk_size")

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=list(separators) if separators else None,
    )
    return [chunk.strip() for chunk in splitter.split_text(content) if chunk.strip()]


def build_seed_question_prompt(
    context: str,
    num_questions: int = 2,
    min_question_tokens: int = 10,
    max_question_tokens: int = 50,
    avoid_keywords: Sequence[str] | None = None,
) -> str:
    """Build the default general-purpose prompt for one text chunk."""
    if num_questions <= 0:
        raise ValueError("num_questions must be greater than zero")
    if min_question_tokens <= 0 or max_question_tokens < min_question_tokens:
        raise ValueError("question token limits must be positive and ordered")

    forbidden = ", ".join(avoid_keywords or []) or "no additional keywords"
    return f"""Generate between 1 and {num_questions} questions that can be fully answered from the context.

Context:
{context}

Rules:
1. Return only a numbered or bulleted list of questions.
2. Each question must be {min_question_tokens} to {max_question_tokens} tokens long.
3. Avoid these keywords: {forbidden}.
4. Every question must be answerable solely from the context.
5. Questions must be clear and unambiguous.
6. Do not use phrases such as 'based on the provided context' or 'according to the context'.
"""


def create_project_llm_generator(model: str | None = None) -> LLMGenerator:
    """Create an LLM callback using the project's configured ``LLMProvider``.

    The returned callback is suitable for ``generate_questions_from_chunks``.
    Configuration and credentials are intentionally resolved only when this
    helper is called, keeping chunking usable without LLM configuration.
    """
    from src.infrastructure.llm_provider import LLMProvider

    provider = LLMProvider(model=model)

    def generate(prompt: str) -> str:
        return provider.get_completion(messages=[{"role": "user", "content": prompt}])

    return generate


def render_prompt(
    prompt: str | PromptBuilder | None,
    context: str,
    num_questions: int,
    min_question_tokens: int,
    max_question_tokens: int,
    avoid_keywords: Sequence[str] | None,
) -> str:
    """Render a caller-provided prompt or the default seed prompt.

    A string prompt may contain ``{context}``, ``{num_questions}``,
    ``{min_question_tokens}``, ``{max_question_tokens}``, and
    ``{avoid_keywords}`` placeholders. A callable receives the same values as
    keyword arguments.
    """
    values = {
        "context": context,
        "num_questions": num_questions,
        "min_question_tokens": min_question_tokens,
        "max_question_tokens": max_question_tokens,
        "avoid_keywords": list(avoid_keywords or []),
    }
    if prompt is None:
        return build_seed_question_prompt(**values)
    if callable(prompt):
        return prompt(**values)
    return prompt.format(**values)


def extract_questions(response: str, max_questions: int | None = None) -> list[str]:
    """Extract clean questions from JSON, numbered, bulleted, or plain output."""
    if not isinstance(response, str) or not response.strip():
        return []

    cleaned = response.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            parsed = parsed.get("questions", parsed.get("output", parsed))
        if isinstance(parsed, list):
            questions = [str(item).strip() for item in parsed if str(item).strip()]
            return questions[:max_questions] if max_questions else questions
    except json.JSONDecodeError:
        pass

    questions = []
    for line in cleaned.splitlines():
        candidate = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if candidate and candidate != line or re.match(r"^\s*\d+[.)]|^\s*[-*]", line):
            if candidate:
                questions.append(candidate)

    if not questions:
        questions = [line.strip() for line in cleaned.splitlines() if line.strip()]

    if len(questions) == 1 and "?" in questions[0]:
        questions = [question.strip() + "?" for question in questions[0].split("?") if question.strip()]

    return questions[:max_questions] if max_questions else questions


def generate_questions_for_chunk(
    chunk: str,
    llm_generate: LLMGenerator,
    chunk_index: int = 0,
    prompt: str | PromptBuilder | None = None,
    num_questions: int = 2,
    min_question_tokens: int = 10,
    max_question_tokens: int = 50,
    avoid_keywords: Sequence[str] | None = None,
) -> list[QuestionRecord]:
    """Generate questions for a single chunk and attach that chunk as context."""
    if not isinstance(chunk, str) or not chunk.strip():
        raise ValueError("chunk must be a non-empty string")

    rendered_prompt = render_prompt(
        prompt=prompt,
        context=chunk,
        num_questions=num_questions,
        min_question_tokens=min_question_tokens,
        max_question_tokens=max_question_tokens,
        avoid_keywords=avoid_keywords,
    )
    response = llm_generate(rendered_prompt)
    return [
        QuestionRecord(question=question, context=chunk, chunk_index=chunk_index)
        for question in extract_questions(response, max_questions=num_questions)
    ]


def generate_questions_from_chunks(
    chunks: Sequence[str],
    llm_generate: LLMGenerator,
    prompt: str | PromptBuilder | None = None,
    num_questions: int = 2,
    min_question_tokens: int = 10,
    max_question_tokens: int = 50,
    avoid_keywords: Sequence[str] | None = None,
) -> list[QuestionRecord]:
    """Generate question/context records for every supplied text chunk."""
    records: list[QuestionRecord] = []
    for chunk_index, chunk in enumerate(chunks):
        records.extend(
            generate_questions_for_chunk(
                chunk=chunk,
                llm_generate=llm_generate,
                chunk_index=chunk_index,
                prompt=prompt,
                num_questions=num_questions,
                min_question_tokens=min_question_tokens,
                max_question_tokens=max_question_tokens,
                avoid_keywords=avoid_keywords,
            )
        )
    return records


def generate_questions_from_content(
    content: str,
    llm_generate: LLMGenerator,
    chunk_size: int = 1600,
    chunk_overlap: int = 200,
    prompt: str | PromptBuilder | None = None,
    num_questions: int = 2,
    min_question_tokens: int = 10,
    max_question_tokens: int = 50,
    avoid_keywords: Sequence[str] | None = None,
) -> list[QuestionRecord]:
    """Run the full text → chunks → question/context-records flow."""
    chunks = chunk_content(
        content=content,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return generate_questions_from_chunks(
        chunks=chunks,
        llm_generate=llm_generate,
        prompt=prompt,
        num_questions=num_questions,
        min_question_tokens=min_question_tokens,
        max_question_tokens=max_question_tokens,
        avoid_keywords=avoid_keywords,
    )


def _demo_llm(_: str) -> str:
    """Deterministic local callback for checking the module without an LLM."""
    return "1. What is the primary purpose described in this text?\n2. Which details support that purpose?"


if __name__ == "__main__":
    demo_content = (
        "A retrieval system stores document chunks as embeddings. "
        "It can retrieve the most relevant chunks when a user submits a question. "
        "Ground-truth datasets pair generated questions with source context."
    )
    demo_records = generate_questions_from_content(
        content=demo_content,
        llm_generate=_demo_llm,
        chunk_size=120,
        chunk_overlap=20,
    )
    print(json.dumps([record.to_dict() for record in demo_records], indent=2))
