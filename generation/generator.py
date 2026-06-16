"""Stage 6 — LLM Generation Layer"""
import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GenerationResult:
    query: str; answer: str; confidence: float
    context_used: str; query_type: str; model_used: str
    token_usage: dict = field(default_factory=dict)


class LLMGenerator:
    SYSTEM_PROMPT = """You are a precise research assistant.
Answer the question using ONLY the provided context.
If context lacks information, say so. Do not fabricate information."""

    def __init__(self, backend="ollama", model_name="phi3",
                 temperature=0.2, max_new_tokens=512, api_base=None):
        self.backend = backend
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.api_base = api_base
        self._pipeline = None

    def generate(self, context) -> GenerationResult:
        prompt = self._prompt(context)
        if self.backend == "openai":
            answer, usage = self._openai(prompt)
        elif self.backend == "huggingface":
            answer, usage = self._hf(prompt)
        elif self.backend == "ollama":
            answer, usage = self._ollama(prompt)
        else:
            answer, usage = self._mock(context)
        confidence = self._confidence(answer, context.context_text)
        return GenerationResult(
            query=context.query, answer=answer, confidence=confidence,
            context_used=context.context_text, query_type=context.query_type,
            model_used=f"{self.backend}/{self.model_name}", token_usage=usage,
        )

    def _prompt(self, ctx) -> str:
        p = f"### Question\n{ctx.query}\n\n### Context\n{ctx.context_text}\n\n"
        p += {"summarization": "Provide a comprehensive summary.\n",
              "reasoning": "Reason step-by-step.\n",
              "multi_hop": "Synthesize information from multiple sections.\n",
              }.get(ctx.query_type, "Give a direct, concise factual answer.\n")
        return p + "\n### Answer\n"

    def _openai(self, prompt):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        kw = {}
        if self.api_base: kw["base_url"] = self.api_base
        if not os.environ.get("OPENAI_API_KEY"): kw["api_key"] = "EMPTY"
        client = OpenAI(**kw)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"system","content":self.SYSTEM_PROMPT},
                      {"role":"user","content":prompt}],
            temperature=self.temperature, max_tokens=self.max_new_tokens,
        )
        ans = resp.choices[0].message.content.strip()
        usage = {"prompt_tokens":resp.usage.prompt_tokens,
                 "completion_tokens":resp.usage.completion_tokens,
                 "total_tokens":resp.usage.total_tokens}
        return ans, usage

    def _hf(self, prompt):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline("text-generation", model=self.model_name,
                                       device_map="auto", max_new_tokens=self.max_new_tokens)
        full = f"[INST] {self.SYSTEM_PROMPT}\n\n{prompt} [/INST]"
        out = self._pipeline(full)[0]["generated_text"]
        return out[len(full):].strip(), {"model": self.model_name}

    def _ollama(self, prompt):
        import urllib.request
        import json
        
        api_url = self.api_base or "http://localhost:11434/api/generate"
        payload = {
            "model": self.model_name,
            "prompt": f"{self.SYSTEM_PROMPT}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_new_tokens
            }
        }
        
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"})
        
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))
                ans = result.get("response", "").strip()
                usage = {
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
                return ans, usage
        except Exception as e:
            return f"Error connecting to Ollama: {e}", {"model": self.model_name}

    def _mock(self, ctx):
        first = ctx.chunks[0].text[:200] if ctx.chunks else "No context."
        answer = (f"[MOCK — configure backend for real answers]\n"
                  f"Type:{ctx.query_type} | Chunks:{len(ctx.chunks)} | Hops:{ctx.retrieval_steps}\n\n"
                  f"Key passage: \"{first[:150]}...\"")
        return answer, {"model":"mock"}

    def _confidence(self, answer, context):
        LOW = ["does not contain","insufficient","unclear","cannot determine","no information"]
        penalty = sum(0.3 for p in LOW if p in answer.lower())
        wc = len(answer.split())
        if wc < 10: penalty += 0.4
        elif wc < 20: penalty += 0.2
        cw = set(context.lower().split())
        aw = set(answer.lower().split())
        if cw and len(aw & cw)/max(len(aw),1) < 0.05:
            penalty += 0.25
        return max(0.0, min(1.0, 1.0-penalty))
