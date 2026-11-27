#!/usr/bin/env python3
"""Diffs ink::render_chat (C++ port) against the real Jinja template (tests/chat_ref.py).

  python3 tests/chat_test.py [--fuzz N] [--seed S] [-v]

Builds tests/bin/chat_test from tests/chat_test.cpp + src/chat.cpp, generates hand-written and
random cases, renders them with both implementations and compares byte for byte.  Error cases must
fail in both; when the template calls raise_exception the message must match exactly.
"""
import argparse
import copy
import json
import os
import random
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BIN = os.path.join(HERE, "bin", "chat_test")

U = {"role": "user", "content": "Hi"}
S = {"role": "system", "content": "You are terse."}
A = {"role": "assistant", "content": "Hello!"}

WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather for a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"enum": ["c", "f"]}},
            "required": ["city"],
        },
    },
}
NESTED = {
    "type": "function",
    "function": {
        "name": "zeta_search",
        "description": "Recherche “avancée” — 日本語 \U0001F600 with \"quotes\" and \\ backslash\n\ttabs",
        "parameters": {
            "type": "object",
            "zzz": 1,
            "aaa": {"b": [1, 2.0, 2.5, True, None, {"y": 1, "x": [[], {}]}], "a": -0.0},
            "properties": {
                "q": {"type": "string", "minLength": 1},
                "limits": {"type": "array", "items": {"type": "number", "minimum": 1e-05, "maximum": 1e16}},
                "é": {"type": "integer", "default": 10000000000},
                "Z": {},
                "a": {"const": 1.5e300},
            },
        },
    },
}


def call(name, args, cid=None, **extra):
    tc = {"type": "function", "function": {"name": name, "arguments": args}}
    if cid is not None:
        tc["id"] = cid
    tc.update(extra)
    return tc


def handwritten():
    cases = []

    def add(name, request, canonicalize=False):
        cases.append({"name": name, "request": request, "canonicalize": canonicalize})

    # --- basic conversations -----------------------------------------------------------------
    add("user only", {"messages": [U]})
    add("system+user", {"messages": [S, U]})
    add("two systems then user", {"messages": [S, {"role": "system", "content": "Second."}, U]})
    add("system only", {"messages": [S]})
    add("no messages", {"messages": []})
    add("messages missing", {})
    add("multi-turn", {"messages": [S, U, A, {"role": "user", "content": "And again?"}, {"role": "assistant", "content": "Sure."}, U]})
    add("system in the middle", {"messages": [U, S, A, S, U]})
    add("assistant first", {"messages": [A, U]})
    add("unicode content", {"messages": [{"role": "user", "content": "¿Qué tal? 日本語 \U0001F600\n\tline2 \"q\" \\ <|end_message|> {{ x }} {% y %}"}]})
    add("whitespace content", {"messages": [{"role": "user", "content": "  \n lead and trail \n\n"}]})

    # --- reasoning effort --------------------------------------------------------------------
    for eff in ["none", "minimal", "low", "medium", "high", "xhigh", "max", " high ", "\thigh\n", " low ", "HIGH", "", " ",
                "bogus", "0.5", 0, 0.0, -0.0, 0.5, 0.99, 0.1, 0.25, 0.3, 0.7, 0.9, 1e-05, 1.5e-05, 0.0001, 0.00012345, 0.123456789012345678,
                1e-300, 1, 1.0, 0.991, -0.1, -1, 100, True, False, None, [], [1], {}, {"a": 1}, 0.30000000000000004, 0.98999999999999999]:
        add(f"effort {eff!r}", {"messages": [S, U], "reasoning_effort": eff})
    add("effort with no messages", {"messages": [], "reasoning_effort": "low"})
    add("effort invalid with system only", {"messages": [S], "reasoning_effort": "nope"})
    add("effort via chat_template_kwargs", {"messages": [U], "chat_template_kwargs": {"reasoning_effort": "low"}}, True)
    add("effort top-level beats kwargs", {"messages": [U], "reasoning_effort": 0.5, "chat_template_kwargs": {"reasoning_effort": "low"}}, True)

    # --- tools -------------------------------------------------------------------------------
    add("one tool", {"messages": [U], "tools": [WEATHER]})
    add("nested unicode tool", {"messages": [S, U], "tools": [NESTED, WEATHER]})
    add("tool without function wrapper", {"messages": [U], "tools": [{"name": "bare", "description": "d", "parameters": {"type": "object"}}]})
    add("tool without wrapper but with type", {"messages": [U], "tools": [{"type": "custom", "name": "bare"}]})
    add("tool missing description", {"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]})
    add("tool missing parameters", {"messages": [U], "tools": [{"type": "function", "function": {"name": "f", "description": "x"}}]})
    add("tool only name", {"messages": [U], "tools": [{"function": {"name": "f"}}]})
    add("tool null fields", {"messages": [U], "tools": [{"type": None, "function": {"name": "f", "description": None, "parameters": None}}]})
    add("tool empty fields", {"messages": [U], "tools": [{"type": "", "function": {"name": "f", "description": "", "parameters": {}}}]})
    add("tool odd-typed fields", {"messages": [U], "tools": [{"type": 7, "function": {"name": 12, "description": 3.0, "parameters": [1, "a"]}}]})
    add("tool extra keys ignored", {"messages": [U], "tools": [{"type": "function", "strict": True, "function": {"name": "f", "strict": True, "extra": {"z": 1}}}]})
    add("tool missing name", {"messages": [U], "tools": [{"type": "function", "function": {"description": "x"}}]})
    add("tool function null", {"messages": [U], "tools": [{"type": "function", "function": None}]})
    add("tool is a string", {"messages": [U], "tools": ["f"]})
    add("tools empty list", {"messages": [U], "tools": []})
    add("tools null", {"messages": [U], "tools": None})
    add("tools with no messages", {"messages": [], "tools": [WEATHER]})
    add("tool_choice none drops tools", {"messages": [U], "tools": [WEATHER], "tool_choice": "none"}, True)
    add("tool_choice auto keeps tools", {"messages": [U], "tools": [WEATHER], "tool_choice": "auto"}, True)
    add("tool key ordering", {"messages": [U], "tools": [{"function": {"name": "k", "parameters": {"b": 1, "a": 2, "B": 3, "A": 4, "_": 5, "é": 6, "z": {"y": 1, "x": 2}, "1": 0, "10": 0, "9": 0}}}]})
    add("tool control chars", {"messages": [U], "tools": [{"function": {"name": "c", "description": "a\u0000b\u0001\u001f\u007f\u0080\b\f\r/ "}}]})

    # --- assistant tool calls ----------------------------------------------------------------
    tc_msgs = [
        U,
        {"role": "assistant", "content": None, "tool_calls": [call("get_weather", {"city": "Paris", "unit": "c"}, "call_1")]},
        {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 21}"},
        {"role": "assistant", "content": "It is 21C."},
    ]
    add("tool call roundtrip by id", {"messages": tc_msgs, "tools": [WEATHER]})
    m = copy.deepcopy(tc_msgs)
    m[2] = {"role": "tool", "name": "get_weather", "content": "{\"temp\": 21}"}
    add("tool result by name", {"messages": m, "tools": [WEATHER]})
    m = copy.deepcopy(tc_msgs)
    m[2]["name"] = "explicit_name"
    add("tool result name beats id", {"messages": m})
    m = copy.deepcopy(tc_msgs)
    m[2]["tool_call_id"] = "unknown"
    add("tool result unknown id", {"messages": m})
    m = copy.deepcopy(tc_msgs)
    del m[2]["tool_call_id"]
    add("tool result no id no name", {"messages": m})
    m = copy.deepcopy(tc_msgs)
    m[2]["name"] = ""
    add("tool result empty name falls back to id", {"messages": m})
    add("string arguments (canonicalized)", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "{\"b\": 1, \"a\": [1, 2.0, {\"z\": null, \"y\": \"é\"}]}", "c1")]}]}, True)
    add("string arguments (not canonicalized)", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "{\"b\": 1}", "c1")]}]})
    add("empty string arguments", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "")]}]})
    add("empty string arguments canonicalized", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "  ")]}]}, True)
    add("invalid JSON arguments", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "{oops")]}]}, True)
    add("string args that parse to a list", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "[1,2]")]}]}, True)
    add("string args that parse to a number", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "42")]}]}, True)
    add("string args that parse to null", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", "null")]}]}, True)
    add("arguments list", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", [1])]}]})
    add("arguments number", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", 5)]}]})
    add("arguments zero", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", 0)]}]})
    add("arguments null", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", None)]}]})
    add("arguments missing", {"messages": [U, {"role": "assistant", "tool_calls": [{"function": {"name": "f"}}]}]})
    add("arguments empty object", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", {})]}]})
    add("argument ordering and nesting", {"messages": [U, {"role": "assistant", "tool_calls": [call("f", {
        "zeta": {"b": 1, "a": {"d": [3, {"k": 1, "j": 2}], "c": 2.0}}, "Alpha": 1, "alpha": 1.0, "_x": None, "été": "ü \"q\" \\ \n",
        "10": True, "9": False, "big": 12345678901234567890, "neg": -7, "f": 1e-07, "g": 1e+22, "h": 0.1, "i": 123456789.125, "empty": [], "eo": {}, "s": ""})]}]})
    add("tool call with content", {"messages": [U, {"role": "assistant", "content": "Let me check.", "tool_calls": [call("get_weather", {"city": "Rome"}, "x")]}]})
    add("tool call with empty content", {"messages": [U, {"role": "assistant", "content": "", "tool_calls": [call("get_weather", {"city": "Rome"}, "x")]}]})
    add("tool call with reasoning", {"messages": [U, {"role": "assistant", "reasoning_content": "Need weather.", "content": "", "tool_calls": [call("get_weather", {"city": "Rome"}, "x")]}]})
    add("tool call with reasoning and content", {"messages": [U, {"role": "assistant", "reasoning_content": "Think.", "content": "Checking", "tool_calls": [call("get_weather", {"city": "Rome"}, "x")]}]})
    add("parallel tool calls + results", {"messages": [
        U,
        {"role": "assistant", "tool_calls": [call("a", {"x": 1}, "1"), call("b", {"y": [1, 2]}, "2")]},
        {"role": "tool", "tool_call_id": "2", "content": "B"},
        {"role": "tool", "tool_call_id": "1", "content": "A"},
        U,
    ]})
    add("duplicate tool call ids", {"messages": [
        U, {"role": "assistant", "tool_calls": [call("a", {}, "1"), call("b", {}, "1")]}, {"role": "tool", "tool_call_id": "1", "content": "r"}]})
    add("tool result id defined later", {"messages": [
        U, {"role": "tool", "tool_call_id": "9", "content": "r"}, {"role": "assistant", "tool_calls": [call("late", {}, "9")]}]})
    add("numeric tool call ids", {"messages": [
        U, {"role": "assistant", "tool_calls": [call("a", {}, 1)]}, {"role": "tool", "tool_call_id": 1.0, "content": "r"}]})
    add("assistant with matching name", {"messages": [U, {"role": "assistant", "name": "f", "content": "x", "tool_calls": [call("f", {"a": 1})]}]})
    add("assistant with mismatching name", {"messages": [U, {"role": "assistant", "name": "g", "tool_calls": [call("f", {"a": 1})]}]})
    add("assistant with name only", {"messages": [U, {"role": "assistant", "name": "persona", "content": "x", "reasoning_content": "r"}]})
    add("user with name (ignored)", {"messages": [{"role": "user", "name": "bob", "content": "x"}]})
    add("tool call name not string", {"messages": [U, {"role": "assistant", "tool_calls": [call(5, {})]}]})
    add("tool call name missing", {"messages": [U, {"role": "assistant", "tool_calls": [{"function": {"arguments": {}}}]}]})
    add("tool call function missing", {"messages": [U, {"role": "assistant", "tool_calls": [{"id": "1"}]}]})
    add("tool call function null", {"messages": [U, {"role": "assistant", "tool_calls": [{"id": "1", "function": None}]}]})
    add("tool call function string", {"messages": [U, {"role": "assistant", "tool_calls": [{"id": "1", "function": "f"}]}]})
    add("tool_calls empty list", {"messages": [U, {"role": "assistant", "content": "", "tool_calls": []}]})
    add("tool_calls null", {"messages": [U, {"role": "assistant", "content": "", "tool_calls": None}]})
    add("tool_calls not a list (number)", {"messages": [U, {"role": "assistant", "tool_calls": 3}]})
    add("tool_calls not a list (string)", {"messages": [U, {"role": "assistant", "tool_calls": "abc"}]})
    add("tool_calls on user ignored", {"messages": [{"role": "user", "content": "x", "tool_calls": [call("f", {})]}]})
    add("id lookup: matching call lacks function", {"messages": [U, {"role": "assistant", "tool_calls": [{"id": "1"}]}, {"role": "tool", "tool_call_id": "1", "content": "r"}]})
    add("id lookup: other calls malformed", {"messages": [
        U, {"role": "assistant", "content": "x", "tool_calls": "zz"}, {"role": "assistant", "content": "x", "tool_calls": {"id": 1}},
        {"role": "tool", "tool_call_id": "1", "content": "r"}]})
    add("id lookup: function name null", {"messages": [
        {"role": "tool", "tool_call_id": "1", "content": "r"}, {"role": "assistant", "content": "x", "tool_calls": [{"id": "1", "function": {"name": None, "arguments": {}}}]}]})
    add("tool name special chars", {"messages": [U, {"role": "assistant", "tool_calls": [call("we\"ird\\né\n", {"k\"\n": "v"})]}]})

    # --- reasoning content -------------------------------------------------------------------
    add("assistant reasoning + content", {"messages": [U, {"role": "assistant", "reasoning_content": "Hmm.", "content": "Answer."}, U]})
    add("assistant reasoning, empty content", {"messages": [U, {"role": "assistant", "reasoning_content": "Hmm.", "content": ""}]})
    add("assistant reasoning, null content", {"messages": [U, {"role": "assistant", "reasoning_content": "Hmm.", "content": None}]})
    add("assistant reasoning only", {"messages": [U, {"role": "assistant", "reasoning_content": "Hmm."}]})
    add("assistant empty reasoning", {"messages": [U, {"role": "assistant", "reasoning_content": "", "content": "A"}]})
    add("assistant null reasoning", {"messages": [U, {"role": "assistant", "reasoning_content": None, "content": "A"}]})
    add("assistant numeric reasoning", {"messages": [U, {"role": "assistant", "reasoning_content": 1.5, "content": "A"}]})
    add("user reasoning ignored", {"messages": [{"role": "user", "reasoning_content": "x", "content": "A"}]})

    # --- content shapes ----------------------------------------------------------------------
    add("assistant empty content", {"messages": [U, {"role": "assistant", "content": ""}]})
    add("assistant null content", {"messages": [U, {"role": "assistant", "content": None}]})
    add("assistant no content key", {"messages": [U, {"role": "assistant"}]})
    add("user empty content", {"messages": [{"role": "user", "content": ""}]})
    add("user null content", {"messages": [{"role": "user", "content": None}]})
    add("user no content", {"messages": [{"role": "user"}]})
    add("system empty content", {"messages": [{"role": "system", "content": ""}, U]})
    add("tool empty content", {"messages": [U, {"role": "tool", "name": "t", "content": ""}]})
    add("content parts text", {"messages": [{"role": "user", "content": [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]}]})
    add("content parts mixed", {"messages": [{"role": "user", "content": [
        "bare", {"type": "input_text", "text": "it"}, {"text": "untyped"}, {"type": "text"}, {"type": "text", "text": 5}, {"type": "text", "text": None}, {}, 7, None, ["x"]]}]})
    add("content parts for assistant with tools", {"messages": [U, {"role": "assistant", "content": [{"type": "text", "text": "a"}], "tool_calls": [call("f", {})]}]})
    add("content parts for system and tool", {"messages": [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]}, U, {"role": "tool", "name": "t", "content": [{"type": "text", "text": "res"}, "more"]}]})
    add("content empty list", {"messages": [{"role": "user", "content": []}]})
    add("content object (iterates keys)", {"messages": [{"role": "user", "content": {"k1": 1, "k2": 2}}]})
    add("content number", {"messages": [{"role": "user", "content": 5}]})
    add("content zero", {"messages": [{"role": "user", "content": 0}]})
    add("content true", {"messages": [{"role": "user", "content": True}]})
    add("content false", {"messages": [{"role": "user", "content": False}]})
    for t in ["image", "input_image", "image_url"]:
        add(f"{t} part for user", {"messages": [{"role": "user", "content": [{"type": "text", "text": "look"}, {"type": t, "image_url": {"url": "data:x"}}]}]})
    add("image part for tool", {"messages": [U, {"role": "tool", "name": "cam", "content": [{"type": "image"}]}]})
    add("image part for assistant", {"messages": [U, {"role": "assistant", "content": [{"type": "image"}]}]})
    add("image part for system", {"messages": [{"role": "system", "content": [{"type": "image_url"}]}]})
    for t in ["audio", "input_audio", "audio_url"]:
        add(f"{t} part", {"messages": [{"role": "user", "content": [{"type": t}]}]})
    add("unknown part type", {"messages": [{"role": "user", "content": [{"type": "video"}]}]})
    add("numeric part type", {"messages": [{"role": "user", "content": [{"type": 3}]}]})
    add("null part type", {"messages": [{"role": "user", "content": [{"type": None, "text": "x"}]}]})
    add("empty part type", {"messages": [{"role": "user", "content": [{"type": "", "text": "x"}]}]})

    # --- role errors -------------------------------------------------------------------------
    for role in ["developer", "function", "", "User", 5, 1.5, None, True]:
        add(f"role {role!r}", {"messages": [U, {"role": role, "content": "x"}]})
    add("role missing", {"messages": [{"content": "x"}]})
    add("role list", {"messages": [{"role": ["user"], "content": "x"}]})
    add("message is a string", {"messages": ["hello"]})
    add("message is null", {"messages": [None]})
    add("error after valid output", {"messages": [S, U, A, {"role": "nope"}]})
    add("effort error precedes role error", {"messages": [U, {"role": "nope"}], "reasoning_effort": "bogus"})
    add("role error precedes effort error", {"messages": [{"role": "nope"}], "reasoning_effort": "bogus"})
    add("tool spec error precedes everything", {"messages": [{"role": "nope"}], "tools": [{}], "reasoning_effort": "bogus"})
    return cases


def fuzz(n, seed):
    rnd = random.Random(seed)
    strings = ["", "x", "Hello", " spaced ", "éè 日本 \U0001F600", "a\"b\\c\nd\te", "<|end_message|>", "0", "user", "f", "g"]
    scalars = strings + [None, True, False, 0, 1, -3, 0.0, 0.5, 1.0, 2.5e-07, 1e+30, 12345678901234]

    def value(depth=0):
        r = rnd.random()
        if depth > 2 or r < 0.55:
            return rnd.choice(scalars)
        if r < 0.78:
            return [value(depth + 1) for _ in range(rnd.randint(0, 3))]
        return {rnd.choice(["a", "b", "B", "_", "é", "k1", "zz", "10", "9", ""]): value(depth + 1) for _ in range(rnd.randint(0, 4))}

    def maybe(d, key, gen, p=0.5):
        if rnd.random() < p:
            d[key] = gen()

    def name():
        return rnd.choice(["f", "g", "get_weather", "", None, "über", 3])

    def tool():
        fn = {}
        maybe(fn, "name", lambda: rnd.choice(["f", "g", "hé", 1]), 0.93)
        maybe(fn, "description", lambda: rnd.choice(strings + [None, 0, 2]))
        maybe(fn, "parameters", lambda: rnd.choice([value(), {"type": "object", "properties": value(1)}]))
        if rnd.random() < 0.75:
            t = {"function": fn}
        else:
            t = fn
        maybe(t, "type", lambda: rnd.choice(["function", "", None, "custom"]))
        return t

    def tool_call():
        fn = {}
        maybe(fn, "name", lambda: rnd.choice(["f", "g", "get_weather", "über", "", 3, None]), 0.95)
        maybe(fn, "arguments", lambda: rnd.choice([{}, value(), {"q": value(), "A": value(), "a": value()}, "{\"a\": 1}", "", None, [], [1], 0]), 0.85)
        tc = {}
        maybe(tc, "function", lambda: fn, 0.96)
        maybe(tc, "id", lambda: rnd.choice(["1", "2", "3", 1, None]), 0.8)
        return tc

    def part():
        r = rnd.random()
        if r < 0.15:
            return rnd.choice(strings)
        p = {}
        maybe(p, "type", lambda: rnd.choice(["text", "input_text", "image", "image_url", "input_image", "audio", "video", "", None, 1]), 0.8)
        maybe(p, "text", lambda: rnd.choice(strings + [None, 4]), 0.8)
        return p

    def content():
        r = rnd.random()
        if r < 0.6:
            return rnd.choice(strings)
        if r < 0.7:
            return None
        if r < 0.95:
            return [part() for _ in range(rnd.randint(0, 3))]
        return rnd.choice([0, 1, True, False, {}, {"k": 1}])

    def message():
        role = rnd.choices(["user", "assistant", "system", "tool", "other", None], weights=[30, 30, 10, 25, 2, 1])[0]
        m = {"role": role}
        maybe(m, "content", content, 0.9)
        if role == "assistant":
            maybe(m, "reasoning_content", lambda: rnd.choice(strings + [None]), 0.4)
            maybe(m, "tool_calls", lambda: rnd.choice([[tool_call() for _ in range(rnd.randint(0, 3))], None, []]), 0.5)
            maybe(m, "name", name, 0.15)
        elif role == "tool":
            maybe(m, "name", name, 0.4)
            maybe(m, "tool_call_id", lambda: rnd.choice(["1", "2", "3", 1, "", None]), 0.7)
        return m

    cases = []
    for i in range(n):
        req = {"messages": [message() for _ in range(rnd.randint(0, 6))]}
        maybe(req, "tools", lambda: [tool() for _ in range(rnd.randint(0, 3))], 0.4)
        maybe(req, "reasoning_effort", lambda: rnd.choice(["none", "low", "high", "max", "x", 0, 0.3, 0.99, 1, None, rnd.random(), rnd.random() * 1e-4]), 0.4)
        maybe(req, "tool_choice", lambda: rnd.choice(["none", "auto"]), 0.2)
        cases.append({"name": f"fuzz {i}", "request": req, "canonicalize": rnd.random() < 0.6})
    return cases


def build():
    srcs = [os.path.join(HERE, "chat_test.cpp"), os.path.join(ROOT, "src", "chat.cpp")]
    if os.path.exists(BIN) and all(os.path.getmtime(BIN) > os.path.getmtime(s) for s in srcs + [os.path.join(ROOT, "src", "chat.h")]):
        return
    os.makedirs(os.path.dirname(BIN), exist_ok=True)
    # chat.cpp references ink::Tokenizer only from the output parser; a stub translation unit is not
    # needed because chat_test.cpp never instantiates the parser (dead-stripped).
    cmd = ["clang++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-I" + os.path.join(ROOT, "vendor"), "-Wl,-dead_strip",
           "-Wl,-undefined,dynamic_lookup", "-o", BIN] + srcs
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuzz", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    build()
    cases = handwritten() + fuzz(args.fuzz, args.seed)
    cases_path = os.path.join(HERE, "bin", "chat_cases.json")
    with open(cases_path, "w", encoding="utf-8") as f:
        json.dump(cases, f, ensure_ascii=False)

    ref = json.loads(subprocess.check_output([sys.executable, os.path.join(HERE, "chat_ref.py"), cases_path]))
    got = json.loads(subprocess.check_output([BIN, cases_path]))
    assert len(ref) == len(got) == len(cases)

    bad = n_ok = n_raised = n_other = 0
    for c, r, g in zip(cases, ref, got):
        same = r["ok"] == g["ok"]
        if same and r["ok"]:
            same = r["text"] == g["text"]
            n_ok += 1
        elif same:
            same = r["raised"] == g["raised"] and (not r["raised"] or r["message"] == g["message"])
            n_raised += r["raised"]
            n_other += not r["raised"]
        if args.verbose and same and not c["name"].startswith("fuzz"):
            print(f"ok   {c['name']}: {(r.get('text') or r.get('message'))!r}")
        if not same:
            bad += 1
            if bad <= 15:
                print(f"MISMATCH {c['name']}\n  request: {json.dumps(c['request'], ensure_ascii=False)}\n  canon:   {c['canonicalize']}\n  jinja:   {r}\n  c++:     {g}")
    print(f"{len(cases)} cases ({len(cases) - args.fuzz} hand-written, {args.fuzz} fuzz): {n_ok} rendered, "
          f"{n_raised} raise_exception, {n_other} other errors; {bad} mismatches")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
