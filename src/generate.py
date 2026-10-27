"""Generate a grounded, cited answer from retrieved chunks an LLM."""
import os
import re
import time

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

NOT_FOUND = "I can't find this in the gov.uk guidance I have."

SYSTEM_PROMPT = f"""You answer questions about UK visas using ONLY the numbered sources provided.

Rules:
- Use only facts stated in the sources. Do not use your own knowledge.
-You may apply a rule from the sources to the person's situation, for example by comparing
their dates, course or circumstances with what the rule requires. Explain that step briefly.
- After each fact, cite its source number in suare brackets, for example [2].
- If the sources do not contain the answer, reply exactly: "{NOT_FOUND}" Do not guess.
- If a rule depends on a date (for example, when someone applies), state each case.
- Keep the answer short: 2 to 5 sentences, in plain English."""

CLOSED_BOOK_PROMPT = f"""You answer questions about UK visas from your own knowledge.

Rules:
- If you are not sure of the answer, reply exactly: "{NOT_FOUND}" Do not guess.
- If a rule depends on a date (for example, when someone applies), state each case.
- Keep the answer short: 2 to 5 sentences, in plain English."""

class AnswerGenerator:
    def __init__(self, min_interval: float = 4.5, max_retries: int =5):
        self.client = OpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.environ["LLM_API_BASE_URL"]
        )
        self.model = os.environ["LLM_MODEL"]
        self.min_interval = min_interval
        self.max_retries = max_retries
        self._last_call = 0.0

    def __call(self, messages: list[dict]) -> str:
        for attempt in range(self.max_retries):
            wait = self.min_interval - (time.time() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.time()
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                )
                return resp.choices[0].message.content.strip()
            except RateLimitError:
                time.sleep(10 * (attempt + 1))
        raise RuntimeError("LLM rate limit: gave up after retries")

    def answer(self, question: str, chunks: list[dict]) -> dict:
        sources = "\n\n".join(f"[{i}] {c['title_path']}\n{c['body']}" for i, c in enumerate(chunks, start=1))
        text = self.__call([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{sources}\n\nQuestion: {question}"},
        ])
        groups = re.findall(r"\[([\d,\s]+)\]", text) 
        numbers = sorted({int(n) for g in groups for n in re.findall(r"\d+", g)})
        citations = [{
            "n": n,
            "chunk_id": chunks[n - 1]["chunk_id"],
            "title_path": chunks[n - 1]["title_path"],
            "url": chunks[n - 1]["url"],
        } for n in numbers if 1 <= n <= len(chunks)]
        return {
            "answer": text,
            "citations": citations,
            "not_found": NOT_FOUND in text
        }

    def answer_without_context(self, question: str) -> dict:
        text = self.__call([
            {"role": "system", "content": CLOSED_BOOK_PROMPT},
            {"role": "user", "content": question},
        ])
        return {"answer": text, "citations": [], "not_found": NOT_FOUND in text}