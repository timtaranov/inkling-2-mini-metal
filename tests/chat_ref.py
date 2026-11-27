#!/usr/bin/env python3
"""Reference renderer for the Inkling 2 chat template, using real jinja2 with the same environment
Hugging Face transformers builds in `_compile_jinja_template`:

  ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[generation tag, loopcontrols])
  tojson = json.dumps(x, ensure_ascii=False, indent=indent, separators=separators, sort_keys=sort_keys)
  raise_exception(message) -> TemplateError

Usage:
  chat_ref.py CASES.json > RESULTS.json     render every case ({"name", "request", "canonicalize"})
  chat_ref.py --check-gguf MODEL.gguf       verify tests/chat_template.jinja equals the GGUF's template
"""
import json
import os
import sys

import jinja2
import jinja2.ext
from jinja2.sandbox import ImmutableSandboxedEnvironment

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_PATH = os.path.join(HERE, "chat_template.jinja")


class RaisedByTemplate(jinja2.exceptions.TemplateError):
    pass


class GenerationExtension(jinja2.ext.Extension):
    """`{% generation %}...{% endgeneration %}`: renders its body (transformers only tracks offsets)."""

    tags = {"generation"}

    def parse(self, parser):
        lineno = next(parser.stream).lineno
        body = parser.parse_statements(["name:endgeneration"], drop_needle=True)
        return jinja2.nodes.CallBlock(self.call_method("_generation_support"), [], [], body).set_lineno(lineno)

    @jinja2.pass_eval_context
    def _generation_support(self, context, caller):
        return caller()


def make_env():
    def raise_exception(message):
        raise RaisedByTemplate(message)

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

    env = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True, extensions=[GenerationExtension, jinja2.ext.loopcontrols]
    )
    env.filters["tojson"] = tojson
    env.globals["raise_exception"] = raise_exception
    return env


def canonicalize(request):
    """Mirror of ink::canonicalize_chat_request."""
    msgs = request.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            if not isinstance(m, dict) or not isinstance(m.get("tool_calls"), list):
                continue
            for tc in m["tool_calls"]:
                fn = tc.get("function") if isinstance(tc, dict) else None
                if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str):
                    continue
                text = fn["arguments"]
                if text.strip(" \t\r\n") == "":
                    fn["arguments"] = {}
                    continue
                parsed = json.loads(text)
                if isinstance(parsed, str):
                    raise ValueError("tool call arguments must be an object")
                fn["arguments"] = parsed
    if "reasoning_effort" not in request:
        kw = request.get("chat_template_kwargs")
        if isinstance(kw, dict) and "reasoning_effort" in kw:
            request["reasoning_effort"] = kw["reasoning_effort"]
    if request.get("tool_choice") == "none":
        request.pop("tools", None)
    if "tools" in request and request["tools"] is not None and not isinstance(request["tools"], list):
        raise ValueError("'tools' must be an array")


def render_case(template, case):
    request = case["request"]
    try:
        if case.get("canonicalize"):
            canonicalize(request)
        kwargs = {}
        for key in ("messages", "tools", "reasoning_effort"):
            if key in request:
                kwargs[key] = request[key]
        return {"ok": True, "text": template.render(**kwargs)}
    except RaisedByTemplate as e:
        return {"ok": False, "raised": True, "message": str(e)}
    except Exception as e:  # noqa: BLE001 - any implicit Jinja/Python error
        return {"ok": False, "raised": False, "message": f"{type(e).__name__}: {e}"}


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--check-gguf":
        from gguf import GGUFReader  # needs PYTHONPATH=<llama.cpp>/gguf-py

        tmpl = GGUFReader(sys.argv[2]).fields["tokenizer.chat_template"].contents()
        same = tmpl == open(TEMPLATE_PATH, encoding="utf-8").read()
        print("template matches GGUF" if same else "TEMPLATE DIFFERS FROM GGUF")
        sys.exit(0 if same else 1)

    template = make_env().from_string(open(TEMPLATE_PATH, encoding="utf-8").read())
    cases = json.load(open(sys.argv[1], encoding="utf-8"))
    results = [dict(render_case(template, c), name=c["name"]) for c in cases]
    json.dump(results, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
