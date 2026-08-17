"""Personalizer: 4-step personalization flow inspired by PAHF.

Flow:
  Step 1 - request triage:        classify()                     → is this a personalization request?
  Step 2 - pre-action interaction: generate_question()             → ask one question before acting
                                   extract_and_store_preference()  → learn from the user's answer
  Step 3 - execution:             (handled by AgentLoop)
  Step 4 - post-action learning:  post_learn()                    → passively extract signals from completed turn
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from pico.tracing import semconv, trace

if TYPE_CHECKING:
    from pico.memory_engine.consolidate.consolidator import MemoryStore
    from pico.providers.base import LLMProvider


# 第 1 步：请求分流，判断是否需要澄清
_CLASSIFY_PROMPT = """\
Classify whether this user request ABSOLUTELY CANNOT be answered without \
knowing a user preference. Default to needs_clarification=false.

## Memory (known preferences)
{memory}

## Recent Conversation
{history}

## User Request
{message}

Reply with JSON only:
{{"needs_clarification": true/false, "domain": "short domain e.g. programming_language"}}

Return needs_clarification=true ONLY when ALL of these are true:
- The request is genuinely ambiguous with no reasonable default
- The missing preference would lead to a COMPLETELY DIFFERENT answer
- Memory does not already contain a relevant preference
- Recent conversation does not already clarify the intent

Return needs_clarification=false if:
- The request is fully specified or has a reasonable default
- Memory already covers the relevant preference
- The preference would not meaningfully change the outcome
- The request is a simple factual, technical, or follow-up question
- The user is responding to or continuing a previous conversation topic

When in doubt, return false — make a reasonable assumption rather than asking."""

# 第 2a 步：行动前交互，生成澄清问题
_QUESTION_PROMPT = """\
Generate ONE short clarifying question for this request.

Request: {message}
Preference domain: {domain}
Known memory: {memory}

Rules:
- Ask only the single most important missing preference
- List concrete options when possible
- Be brief and natural

Output the question line only, in this exact format:
Quick question: [question]? Options: [A] / [B] / [C]"""

# 第 2b 步：行动前交互，从用户回答中提取偏好
_EXTRACT_PROMPT = """\
Extract reusable preference facts from this clarification Q&A.

Original request: {original_message}
Question asked: {question}
User answer: {answer}

Output JSON:
{{"facts": ["User always uses Python for scripts"], "section": "Preferences"}}

Rules:
- State facts as general reusable rules, NOT tied to this specific task
- Only include facts useful for future interactions
- 1-3 facts maximum
- If the answer reveals no reusable preference, return {{"facts": [], "section": "Preferences"}}"""

# 第 4 步：行动后学习，从已完成交互中被动提取信号
_POST_LEARN_PROMPT = """\
Analyze this completed interaction for new preference signals.

Request: {message}
Response summary: {response_summary}

## Current Memory
{memory}

Did this interaction reveal NEW preferences not already in memory?

Output JSON:
{{"has_new_preference": false, "new_facts": [{{"text": "...", "category": "preference"}}]}}

Each fact object has:
- text: the reusable rule statement about the user
- category: "preference" for general preferences such as languages, tools,
  communication style, and topics

Examples of "preference" category:
  - "User prefers Python for scripting tasks"
  - "User prefers concise responses without preamble"

Only set has_new_preference=true for facts that are:
- Genuinely new (not already in memory)
- Reusable (will help future interactions)
- Expressible as general rules about the user

Legacy format (flat list of strings with a `section` field) is still accepted
for backwards compatibility — category defaults to "preference" in that case."""


class Personalizer:
    """Implements the 4-step PAHF-inspired personalization flow.

    All methods are safe to call independently — failures are logged
    and return neutral defaults so the main agent loop is never blocked.
    """

    def __init__(self, memory: MemoryStore, provider: LLMProvider, model: str):
        self.memory = memory
        self.provider = provider
        self.model = model

    @trace.instrument("personalize.classify", kind="memory", extract=semconv.personalize)
    async def classify(self, message: str, history: list[dict] | None = None) -> dict:
        """Determine if the request needs a personalization clarification question.

        Args:
            message: The current user message.
            history: Recent conversation history (last 2-4 messages) for context.

        Returns:
            {"needs_clarification": bool, "domain": str}
            Falls back to {"needs_clarification": False} on any error.
        """
        current_memory = self.memory.read_long_term()

        history_text = self._format_history(history) if history else "(no prior context)"

        prompt = _CLASSIFY_PROMPT.format(
            memory=current_memory or "(empty)",
            history=history_text,
            message=message,
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,  # 分类需要确定性输出
                max_tokens=100,  # JSON 很短，限制令牌数以节省成本
            )
            result = self._parse_json(
                response.content or "",
                fallback={"needs_clarification": False, "domain": ""},
            )
            logger.debug("Personalizer.classify: {}", result)
            return result
        except Exception:
            logger.exception("Personalizer.classify failed, skipping clarification")
            return {"needs_clarification": False, "domain": ""}

    @trace.instrument("personalize.question", kind="memory", extract=semconv.personalize)
    async def generate_question(self, message: str, domain: str) -> str:
        """Generate a single focused clarifying question for the given request.

        Returns:
            Question string, e.g. "Quick question: Which language? Options: Python / Go"
            Returns "" on failure so the caller can skip clarification gracefully.
        """
        current_memory = self.memory.read_long_term()

        prompt = _QUESTION_PROMPT.format(
            message=message,
            domain=domain,
            memory=current_memory or "(empty)",
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,  # 少量随机性让问题更自然
                max_tokens=120,
            )
            question = (response.content or "").strip()
            logger.debug("Personalizer.generate_question: {}", question)
            return question
        except Exception:
            logger.exception("Personalizer.generate_question failed, skipping clarification")
            return ""

    @trace.instrument("personalize.extract", kind="memory", extract=semconv.personalize)
    async def extract_and_store_preference(self, original_message: str, question: str, answer: str) -> bool:
        """Extract reusable preferences from a Q&A pair and persist to MEMORY.md.

        Called after the user answers a clarifying question.
        Returns True if at least one fact was stored.
        """
        prompt = _EXTRACT_PROMPT.format(
            original_message=original_message,
            question=question,
            answer=answer,
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=200,
            )
            result = self._parse_json(
                response.content or "",
                fallback={"facts": [], "section": "Preferences"},
            )

            facts: list[str] = result.get("facts", [])
            section: str = result.get("section", "Preferences")

            if not facts:
                logger.debug("Personalizer.extract: no reusable preference found in Q&A")
                return False

            self._append_to_memory_section(section, facts)
            logger.info("Personalizer.extract: stored {} fact(s) → {}", len(facts), facts)
            return True

        except Exception:
            logger.exception("Personalizer.extract_and_store_preference failed")
            return False

    @trace.instrument("personalize.postlearn", kind="memory", extract=semconv.personalize)
    async def post_learn(self, message: str, response_summary: str) -> bool:
        """Passively extract new preference signals from a completed interaction.

        Intended to run as a background asyncio task — never blocks the response.
        Returns True if new facts were stored.

        Accepts both the new per-fact ``{text, category}`` schema and the
        legacy flat-list format (backwards-compatible; legacy facts default
        to the "preference" category).
        """
        current_memory = self.memory.read_long_term()

        prompt = _POST_LEARN_PROMPT.format(
            message=message,
            response_summary=response_summary[:600],  # 避免提示无限增长
            memory=current_memory or "(empty)",
        )

        try:
            response = await self.provider.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.0,
                max_tokens=250,
            )
            result = self._parse_json(
                response.content or "",
                fallback={"has_new_preference": False, "new_facts": []},
            )

            if not result.get("has_new_preference"):
                return False

            grouped = self._group_facts_by_category(
                result.get("new_facts", []),
                legacy_section=result.get("section", "Preferences"),
            )
            if not grouped:
                return False

            total = 0
            for section, facts in grouped.items():
                self._append_to_memory_section(section, facts)
                total += len(facts)
                logger.info(
                    "Personalizer.post_learn: stored {} fact(s) → section={} facts={}",
                    len(facts),
                    section,
                    facts,
                )
            return total > 0

        except Exception:
            logger.exception("Personalizer.post_learn failed")
            return False

    _CATEGORY_SECTION_MAP = {
        "preference": "Preferences",
    }

    @classmethod
    def _group_facts_by_category(cls, new_facts: list, legacy_section: str = "Preferences") -> dict[str, list[str]]:
        """Normalize the two possible ``new_facts`` shapes into
        ``{section_header: [fact_text, ...]}``.

        - New shape: ``[{"text": str, "category": str}]``
        - Legacy shape: ``["fact text", "fact text"]`` with a sibling
          ``section`` field on the result object (passed as ``legacy_section``).
        """
        grouped: dict[str, list[str]] = {}
        for item in new_facts or []:
            if isinstance(item, dict):
                text = str(item.get("text", "")).strip()
                category = str(item.get("category", "preference")).strip().lower()
                section = cls._CATEGORY_SECTION_MAP.get(category, legacy_section)
            elif isinstance(item, str):
                text = item.strip()
                section = legacy_section
            else:
                continue
            if not text:
                continue
            grouped.setdefault(section, []).append(text)
        return grouped

    @staticmethod
    def _format_history(history: list[dict], max_messages: int = 4) -> str:
        """Format recent conversation history into a compact string for prompts."""
        recent = history[-max_messages:]
        lines = []
        for m in recent:
            role = m.get("role", "unknown").upper()
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                text = content[:200] + "..." if len(content) > 200 else content
                lines.append(f"{role}: {text}")
        return "\n".join(lines) if lines else "(no prior context)"

    def _parse_json(self, text: str, fallback: dict) -> dict:
        """Extract and parse the first JSON object found in text.

        Robust to LLM wrapping the JSON in extra prose.
        """
        start = text.find("{")
        end = text.rfind("}") + 1
        if start < 0 or end <= start:
            return fallback
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return fallback

    def _append_to_memory_section(self, section: str, facts: list[str]) -> None:
        """Append new facts under the given section header in MEMORY.md.

        - If the section already exists: inserts lines right after the header.
        - If the section is missing: appends a new section at the end of the file.

        Read-modify-write is fcntl-locked via ``MemoryStore.locked()`` so
        concurrent MemoryConsolidator writers on
        another process don't clobber the update.
        """
        header = f"## {section}"
        fact_lines = "\n".join(f"- {f}" for f in facts)

        with self.memory.locked():
            current = self.memory.read_long_term()
            if header in current:
                updated = current.replace(header, f"{header}\n{fact_lines}", 1)
            else:
                updated = current.rstrip() + f"\n\n{header}\n{fact_lines}\n"
            self.memory.write_long_term(updated)
