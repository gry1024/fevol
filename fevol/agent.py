"""轻量 LLM 客户端与演化代理：推理、演化、策略反思。"""
import json
import os
import re
import time
import urllib.error
import urllib.request

from .prompts import DEFAULT_STRATEGY, EVOLVE, REASON, REFLECT, SYSTEM, dump


class LLMError(RuntimeError):
    pass


class LLM:
    def __init__(self, config, api_key=None):
        self.cfg = config
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise LLMError("缺少 DEEPSEEK_API_KEY 环境变量")

    def chat(self, prompt, system=SYSTEM):
        """单轮对话，返回纯文本；网络错误按指数退避重试。"""
        body = json.dumps({
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "temperature": self.cfg.temperature,
            "max_tokens": 4096,
            "response_format": {"type": "json_object"},
        }).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        last = None
        for attempt in range(self.cfg.retries):
            request = urllib.request.Request(self.cfg.base_url, body, headers)
            try:
                with urllib.request.urlopen(request, timeout=self.cfg.timeout) as response:
                    payload = json.load(response)
                return payload["choices"][0]["message"]["content"]
            except (urllib.error.URLError, KeyError, json.JSONDecodeError, TimeoutError) as exc:
                last = exc
                time.sleep(min(2 ** attempt, 8))
        raise LLMError(f"LLM 调用失败: {last}")


def parse_json(text):
    """容忍代码块包裹与多余前缀，取出第一个 JSON 对象。"""
    text = re.sub(r"^```(?:json)?|```$", "", (text or "").strip(), flags=re.M).strip()
    start = text.find("{")
    if start < 0:
        raise ValueError("响应中没有 JSON 对象")
    for end in range(len(text), start, -1):
        if text[end - 1] == "}":
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    raise ValueError("JSON 解析失败")


class EvolveAgent:
    def __init__(self, llm, config):
        self.llm, self.cfg = llm, config
        self.strategy = DEFAULT_STRATEGY

    def _ask(self, prompt, validate):
        error = ""
        for _ in range(2):
            message = prompt if not error else f"{prompt}\n上次输出无效：{error}\n请只返回符合结构要求的 JSON。"
            try:
                return validate(parse_json(self.llm.chat(message)))
            except (ValueError, TypeError, KeyError) as exc:
                error = str(exc)
        raise LLMError(f"代理输出反复无效: {error}")

    def reason(self, parent, reward, nodes, node_list):
        prompt = REASON.format(parent=parent, reward=reward, nodes=nodes,
                               nodes_list="\n".join(node_list),
                               max_mechanisms=self.cfg.max_mechanisms,
                               strategy=self.strategy)

        def validate(data):
            understanding = data.get("understanding")
            mechanisms = data.get("mechanisms")
            if not isinstance(understanding, str) or not understanding.strip():
                raise ValueError("understanding 必须是非空字符串")
            if not isinstance(mechanisms, list):
                raise ValueError("mechanisms 必须是列表")
            cleaned = []
            for item in mechanisms[: self.cfg.max_mechanisms]:
                if not isinstance(item, dict):
                    raise ValueError("mechanism 必须是对象")
                path, replace = item.get("path", []), item.get("replace")
                if not isinstance(path, list) or not all(isinstance(p, int) for p in path):
                    raise ValueError("path 必须是整数列表")
                if not isinstance(replace, str) or not replace.strip():
                    raise ValueError("replace 必须是非空表达式")
                cleaned.append({"path": path, "replace": replace.strip(),
                                "hypothesis": str(item.get("hypothesis", "")).strip()})
            return {"understanding": understanding.strip(), "mechanisms": cleaned}

        return self._ask(prompt, validate)

    def evolve(self, parent, abs_reward, cost, nodes, understanding, evidence, memory,
               existing):
        prompt = EVOLVE.format(parent=parent, abs_reward=abs_reward, cost=cost, nodes=nodes,
                               understanding=understanding, evidence=dump(evidence),
                               memory=dump(memory), existing=dump(existing),
                               max_children=self.cfg.max_children, strategy=self.strategy)

        def validate(data):
            updated = data.get("updated_understanding")
            offspring = data.get("offspring")
            if not isinstance(updated, str) or not updated.strip():
                raise ValueError("updated_understanding 必须是非空字符串")
            if not isinstance(offspring, list):
                raise ValueError("offspring 必须是列表")
            cleaned = []
            for item in offspring[: self.cfg.max_children]:
                if not isinstance(item, dict) or not isinstance(item.get("expression"), str):
                    raise ValueError("offspring 项必须含 expression")
                if not item["expression"].strip():
                    raise ValueError("expression 不能为空")
                cleaned.append({"expression": item["expression"].strip(),
                                "rationale": str(item.get("rationale", "")).strip()})
            return {"updated_understanding": updated.strip(), "offspring": cleaned}

        return self._ask(prompt, validate)

    def reflect(self, records):
        prompt = REFLECT.format(strategy=self.strategy, records=dump(records))

        def validate(data):
            strategy = data.get("strategy")
            if not isinstance(strategy, str) or not strategy.strip():
                raise ValueError("strategy 必须是非空字符串")
            return strategy.strip()[:400]

        self.strategy = self._ask(prompt, validate)
        return self.strategy
