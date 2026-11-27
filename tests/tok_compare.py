#!/usr/bin/env python3
"""Compare ink::Tokenizer (tests/tok_test) against llama.cpp's llama-tokenize on a generated corpus.

Every sample is tokenized by both, with and without special-token parsing, and the id lists must be
identical.  For valid UTF-8 samples decode(encode(x)) == x is checked as well.

usage: tok_compare.py --tok-test BIN --ref-bin llama-tokenize --ref-model ref.gguf [--model our.gguf]
                      [--llama-src DIR] [--work DIR] [--random N] [--seed S] [--jobs J]
"""
import argparse
import concurrent.futures as cf
import glob
import os
import random
import subprocess
import sys
import tempfile

SPECIALS_USED = ["<|endoftext|>", "<|message_user|>", "<|content_text|>", "<|end_message|>"]


# ----------------------------------------------------------------------------------------------- corpus

def handwritten(specials):
    s = {}
    s["empty"] = ""
    for i, ch in enumerate(["a", "Z", " ", "\n", "\r", "\t", "7", "'", "/", "é", "中", "🙂", "\u0301", "\u00a0",
                            "\u2028", "\x00", "\x7f", "<", "|", "\ufffd", "\U0001F468", "ß", "İ"]):
        s["single_%02d" % i] = ch
    s["prose"] = (
        "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of "
        "foolishness, it was the epoch of belief, it was the epoch of incredulity. Mr. O'Neil's co-operative "
        "re-entered the U.S.A. market in 1999; e.g. NASA, UNESCO and the iPhone/iPad makers disagreed.\n\n"
        "\"Well,\" she said -- half-smiling -- \"that's that.\" Then: silence... (Or was it?) [sic] {ok} <tag> "
        "naïve café résumé coöperate Zürich Ångström façade jalapeño São Paulo Łódź\n")
    s["contractions"] = (
        "I'm you're he's she'd we'll they've don't can't won't it's that'd who'll y'all'd've\n"
        "I'M YOU'RE HE'S SHE'D WE'LL THEY'VE DON'T CAN'T WON'T IT'S THAT'D WHO'LL\n"
        "i'M yOU'rE hE'S sHE'd wE'lL tHEY'Ve dON'T 's 't 're 've 'm 'll 'd 'S 'T 'RE 'VE 'M 'LL 'D\n"
        "x's X'S xX'sS 'sup 'tis 'twas rock'n'roll O'Brien d'Artagnan l'homme qu'il 5's 5'S a'1 a'' ''a 'a' "
        "é's É'S д'd 中's a'ré a'lé a'vé a'l a'r a'v a' ' '' ''' a'sb a'llo a'l l\n")
    s["case_runs"] = (
        "camelCase PascalCase snake_case SCREAMING_SNAKE kebab-case HTTPServer XMLHttpRequest iOS macOS "
        "aB Ab AB ab aBc AbC ABc abC ABCd aBCD éA Aé ÉA AÉ aÉ Éa ÉÉ éé AéB aéB AÉb HELLOé éHELLO HELLOéWORLD "
        "ÀB Bà ÀÀb bÀÀ ǅ ǆ Ǆ ǅa aǅ ʰa aʰ Aʰ ʰA ΑΒΓαβγ ΑβΓ αΒγ ДОМдом домДОМ ДомДом\n")
    s["numbers"] = (
        "0 1 12 123 1234 12345 123456 1234567 12345678 123456789 1234567890 3.14159 -42 +7 1e10 0x1F 0b1010 "
        "1,234,567.89 12:34:56 2024-01-15T10:30:00Z 192.168.1.1 555-1234 (555) 123-4567 ½ ¾ ² ³ ¹ ٠١٢٣٤٥٦٧٨٩ "
        "०१२३४५६७८९ ⅧⅨ ①②③ 一二三 １２３４５ 1st 2nd 3rd 4th a1b2c3 x86_64 v1.2.3-rc4 99% $100 €50 #1 "
        + "7" * 100 + " " + "0123456789" * 30 + "\n" + "9" * 1000 + "\n")
    s["whitespace"] = (
        "a b  c   d    e     f      g\n\nh\n\n\ni\r\nj\r\n\r\nk\tl\t\tm \t n\t \to \n p\n q \n\n r  \n  s\n"
        " \n \n  \n\t\n\r\n \r\n\r \r\r\n\n\r x\v\fy \v z\f\n" + " " * 100 + "end" + "\n" * 50 + "\t" * 40 + "x"
        + " \t" * 30 + "\n" + "  \n" * 20 + "tail   ")
    s["whitespace_only_spaces"] = " " * 37
    s["whitespace_only_newlines"] = "\n" * 17
    s["whitespace_only_mixed"] = " \t\n\r \n\t  \r\n   "
    s["whitespace_trailing"] = "abc   "
    s["whitespace_leading"] = "   abc"
    s["whitespace_unicode"] = (
        "a\u00a0b\u2003c\u2009d\u3000e\u2028f\u2029g\u0085h\u1680i\u202fj\u205fk \u00a0 l\u00a0\u00a0m\u200bn"
        "\u2060o\ufeffp \u3000\n\u00a0\n x\u00a0\u00a0\u00a0y\u2003\u2003z\u00a0")
    s["punctuation"] = (
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ !!! ??? ... --- === *** ### /// \\\\\\ ::: ;;; <<< >>> ||| &&& "
        "a/b a//b a///b /a/ ./x ../y http://example.com/path/to?x=1&y=2#frag ->  =>  ==  !=  <=  >=  ++  --  "
        "«quoted» “smart” ‘single’ – — … • · † ‡ § ¶ © ® ™ ° ± × ÷ ¿¡ 、。「」『』【】《》 ！？\n"
        "x/\ny/\r\nz//\n\n/ /\n a /\n,/\n ,/\n\n  ./\n" + "=" * 80 + "\n" + "-" * 200 + "\n" + "*" * 33 + "\n"
        + "#" * 500 + "\n")
    s["json"] = (
        '{\n  "name": "Inkling",\n  "version": 2,\n  "tags": ["a", "b", "c"],\n  "nested": {"x": 1.5e-3, '
        '"y": null, "z": true, "w": false},\n  "str": "line1\\nline2\\t\\"quoted\\" \\u00e9",\n  "arr": '
        '[[1,2,3],[4,5,6],[]],\n  "empty": {},\n  "id": "550e8400-e29b-41d4-a716-446655440000"\n}\n'
        '{"a":1,"b":[true,false,null],"c":{"d":"e"}}\n[{"role":"user","content":"hi"},{"role":"assistant",'
        '"content":"hello"}]\n')
    s["markdown"] = (
        "# Title\n\n## Subtitle\n\n* item 1\n* item 2\n  * nested\n\n1. first\n2. second\n\n> quote\n>> nested "
        "quote\n\n```python\ndef f(x):\n    return x ** 2  # square\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
        "**bold** _italic_ ~~strike~~ `code` [link](http://x.y/z) ![img](a.png)\n\n---\n\n- [ ] todo\n- [x] done\n"
        "\n<details><summary>x</summary>\n\ntext\n\n</details>\n\nFootnote[^1].\n\n[^1]: The note.\n")
    s["code_misc"] = (
        "#include <stdio.h>\nint main(int argc, char **argv) {\n\tfor (int i = 0; i < argc; ++i) {\n\t\tprintf(\"%d: "
        "%s\\n\", i, argv[i]);\n\t}\n\treturn 0;\n}\n\n"
        "fn main() -> Result<(), Box<dyn Error>> {\n    let v: Vec<u32> = (0..10).map(|x| x * 2).collect();\n    "
        "println!(\"{:?}\", v);\n    Ok(())\n}\n\n"
        "SELECT a.id, COUNT(*) AS n FROM t1 a LEFT JOIN t2 b ON a.id = b.a_id WHERE a.x >= 10 GROUP BY 1;\n\n"
        "const f = async ({a, b = 2, ...rest}) => { await g?.(a ?? b); return `${a}-${b}`; };\n\n"
        "<div class=\"x\" id='y'><!-- c --><a href=\"#\">link</a><br/></div>\n\n"
        "for f in *.txt; do echo \"$f\" | sed -e 's/a/b/g' >> out.log 2>&1 || exit 1; done\n\n"
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n\tch := make(chan int, 1)\n\tgo func() { ch <- 42 }()\n\t"
        "fmt.Println(<-ch)\n}\n\n"
        "class A(B):\n    def __init__(self, *args, **kwargs):\n        super().__init__(*args, **kwargs)\n        "
        "self._x: dict[str, list[int]] = {}\n\n    @property\n    def x(self): return self._x\n")
    s["cjk"] = (
        "今天天氣很好，我們去公園散步吧。日本語のテキストです。カタカナとひらがなと漢字。한국어 텍스트입니다. "
        "简体中文测试：你好，世界！这是一个测试。中文English混合text测试123数字。ｆｕｌｌｗｉｄｔｈ　ＴＥＸＴ。"
        "𠀀𠀁𠀂 㐀㐁 ﾊﾝｶｸ ㈱ ㍿ ヿ ゟ 々 〆 〇\n")
    s["cyrillic_greek"] = (
        "Съешь же ещё этих мягких французских булок, да выпей чаю. ПРИВЕТ мир! Привет, МИР! Ёлка ёж. "
        "Το γρήγορο καφέ αλεπού πηδάει πάνω από το τεμπέλικο σκυλί. ΑΘΗΝΑ Αθήνα. Їжак ґанок єнот.\n")
    s["arabic_hebrew"] = (
        "مرحبا بالعالم! هذا نص عربي للاختبار. اللُّغَةُ العَرَبِيَّةُ جَمِيلَةٌ. ١٢٣٤٥ فارسی: سلام دنیا، این یک آزمایش است.\n"
        "שלום עולם! זהו טקסט בעברית. בְּרֵאשִׁית בָּרָא אֱלֹהִים. اردو: یہ ایک ٹیسٹ ہے۔\n")
    s["indic_thai"] = (
        "नमस्ते दुनिया! यह हिंदी में एक परीक्षण है। क्षत्रिय ज्ञान श्री कृष्ण। हिन्दी की मात्राएँ: कि की कु कू के कै को कौ कं कः\n"
        "বাংলা লেখা পরীক্ষা। தமிழ் உரை சோதனை. తెలుగు పరీక్ష. ಕನ್ನಡ ಪರೀಕ್ಷೆ. മലയാളം പരീക്ഷ. ગુજરાતી. ਪੰਜਾਬੀ.\n"
        "สวัสดีชาวโลก นี่คือการทดสอบภาษาไทย ກຂຄ မြန်မာစာ ខ្មែរ བོད་ཡིག සිංහල\n")
    s["combining"] = (
        "e\u0301 a\u0300\u0301\u0302 n\u0303o \u0301abc abc\u0301 ABC\u0301 \u0301ABC A\u0301B a\u0301\u0301B "
        "\u0301 \u0301\u0301 .\u0301 !\u0301a 1\u0301 1\u03012 \u0301' \u0301's a\u0301's Z\u0351\u0352a\u0353 "
        "t\u0336e\u0336x\u0336t\u0336 \u20dd \u0489 \n\u0301\n \u0301 x \u200d\u0301 \u0301\u200d /\u0301/ "
        "..\u0301.. ..\u0301a \u0301.. \u0301\u0301AB \u0301\u0301ab AB\u0301\u0301 \u0301A\u0301\n")
    s["emoji"] = (
        "😀 😃 😄 👍 👍🏽 👋🏻 🙋🏿‍♀️ 👨‍👩‍👧‍👦 👩‍💻 🏳️‍🌈 🏴‍☠️ 🇺🇸🇯🇵🇩🇪 ❤️ ♥︎ ☺️ ©️ #️⃣ 1️⃣ 🔟 "
        "🧑🏾‍🤝‍🧑🏼 👁️‍🗨️ text🙂text 🙂text text🙂 🙂🙂🙂 a😀b😀c ⌚ ⭐ ✅ ➡️ ⬆ 🀄 🃏 🎉🎊 🤷‍♂️ 🫠 🪿\n")
    s["mixed_scripts"] = (
        "Hello世界Привет مرحبا🙂नमस्ते123abcДЕF中a文b字c ΑλφαBeta日本語Englishрусский café中文naïve "
        "x²+y²=z² ∑∫√∞≠≤≥ α→β ⇒ ℝℕℤ 𝔘𝔫𝔦𝔠𝔬𝔡𝔢 𝓼𝓬𝓻𝓲𝓹𝓽 𝟘𝟙𝟚 ℃ ㎏ № ™ ﬁ ﬂ ǆ ß ẞ ı İ ſ K Å\n")
    s["control"] = (
        "a\x00b\x01c\x02\x03\x04\x05\x06\x07\x08d\x0b\x0c\x0e\x0f\x10\x1a\x1b[0m\x1c\x1d\x1e\x1f\x7f e"
        "\u0080\u0081\u0085\u008d\u009f\u00ad\u200b\u200c\u200d\u200e\u200f\u2060\ufeff\ufff9\ufffd\uffff"
        "\U000e0001\U000e0041\U0010ffff\ue000\uf8ff\U000f0000 f\x00\x00\x00 \x00 \x1b\x1b g")
    s["special_basic"] = "<|message_user|><|content_text|>hi<|end_message|>"
    s["special_chat"] = (
        "<|message_user|><|content_text|>What's 2+2?<|end_message|>\n<|message_user|> <|content_text|> spaced "
        "<|end_message|> \n\n<|endoftext|><|endoftext|> <|endoftext|>x<|endoftext|>\n")
    s["special_lookalike"] = (
        "<|message_user <|unknown|> <|message_user|x <|message_user| > < |message_user|> <|MESSAGE_USER|> "
        "<|message_user|message_user|> <|<|message_user|>|> <<|end_message|>> <|end_message|<|end_message|> "
        "|message_user|> <| |> <||> <|endoftext |> <|endoftext|\n<|unused_200007|> <|unused_1|> <|unused_|>")
    s["special_all"] = " ".join(specials) + "\n" + "".join(specials) + "\n" + "x".join(reversed(specials))
    s["special_edges"] = "".join(
        sp[:k] + " " for sp in specials[:6] for k in range(1, len(sp))) + "".join(
        sp[k:] + "\n" for sp in specials[:6] for k in range(1, len(sp)))
    s["special_adjacent_text"] = (
        "abc<|end_message|>def  <|end_message|>  ghi\n<|end_message|>\njkl 123<|end_message|>456 "
        "can<|end_message|>'t é<|end_message|>\u0301 \n \n<|end_message|> \n \n")
    s["long_word"] = "a" * 3000 + " " + "ab" * 700 + " " + "Zq" * 500 + " " + "é" * 400 + " " + "supercalifragilistic" * 40
    s["long_punct"] = "!?" * 900 + " " + "<|" * 300 + "|>" * 300 + "'s" * 300 + "/" * 700
    s["base64ish"] = (
        "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24sIGJ1dCBieSB0aGlzIHNpbmd1bGFyIHBhc3Npb24gZnJv"
        "bSBvdGhlciBhbmltYWxzLCB3aGljaCBpcyBhIGx1c3Qgb2YgdGhlIG1pbmQ= deadbeefCAFEBABE0123456789abcdefABCDEF "
        "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855 "
        "AbCdEfGhIjKlMnOpQrStUvWxYz aBcDeFgHiJkLmNoPqRsTuVwXyZ QWERTYUIOPasdfghjklZXCVBNM qwertyuiopASDFGHJKL")
    return {k: v.encode("utf-8") for k, v in s.items()}


def invalid_utf8():
    s = {}
    s["inv_lone_cont"] = b"a\x80b\xbfc"
    s["inv_truncated"] = b"abc\xe4\xb8"
    s["inv_trunc2"] = b"\xc3"
    s["inv_trunc4"] = b"x\xf0\x9f\x98"
    s["inv_overlong"] = b"\xc0\x80 \xc1\xbf \xe0\x80\x80 \xf0\x80\x80\x80 \xe0\x9f\xbf"
    s["inv_surrogate"] = b"a\xed\xa0\x80b\xed\xbf\xbf c\xed\xa0\xbd\xed\xb8\x80"
    s["inv_too_big"] = b"\xf4\x90\x80\x80 \xf5\x80\x80\x80 \xf8\x88\x80\x80\x80 \xfc\x84\x80\x80\x80\x80 \xfe \xff"
    s["inv_mixed"] = "héllo wörld ".encode("utf-8") + b"\xff\xfe" + "中文".encode("utf-8")[:-1] + b" ok \xe9t\xe9 latin1 caf\xe9's"
    s["inv_all_bytes"] = bytes(range(256))
    s["inv_all_bytes_rev"] = bytes(reversed(range(256))) * 2
    s["inv_special"] = b"\xff<|end_message|>\xc3<|end_message|>\xa9"
    return s


RANGES = {
    "ascii_lower": [(0x61, 0x7A)], "ascii_upper": [(0x41, 0x5A)], "digits": [(0x30, 0x39)],
    "ascii_punct": [(0x21, 0x2F), (0x3A, 0x40), (0x5B, 0x60), (0x7B, 0x7E)],
    "space": [(0x20, 0x20)], "ws": [(0x09, 0x0D), (0x20, 0x20)], "controls": [(0x00, 0x1F), (0x7F, 0x9F)],
    "latin1": [(0xA0, 0xFF)], "latin_ext": [(0x100, 0x24F)], "ipa_mod": [(0x250, 0x2FF)],
    "combining": [(0x300, 0x36F), (0x1AB0, 0x1AFF), (0x20D0, 0x20FF), (0xFE00, 0xFE0F)],
    "greek": [(0x370, 0x3FF)], "cyrillic": [(0x400, 0x52F)], "armenian_hebrew": [(0x530, 0x5FF)],
    "arabic": [(0x600, 0x6FF), (0x750, 0x77F)], "devanagari": [(0x900, 0x97F)], "indic": [(0x980, 0xDFF)],
    "thai_lao_tibetan": [(0xE00, 0xFFF)], "georgian_hangul_jamo": [(0x10A0, 0x11FF)],
    "ethiopic_etc": [(0x1200, 0x18AF)], "latin_add": [(0x1E00, 0x1FFF)],
    "gen_punct": [(0x2000, 0x206F)], "super_currency": [(0x2070, 0x20CF)], "letterlike_numforms": [(0x2100, 0x218F)],
    "arrows_math": [(0x2190, 0x23FF)], "boxes_misc": [(0x2400, 0x27BF)], "sym_misc": [(0x27C0, 0x2BFF)],
    "cjk_sym_kana": [(0x3000, 0x30FF)], "cjk_ext": [(0x3100, 0x33FF)], "cjk": [(0x4E00, 0x9FFF)],
    "cjk_a": [(0x3400, 0x4DBF)], "yi_etc": [(0xA000, 0xABFF)], "hangul": [(0xAC00, 0xD7A3)],
    "private": [(0xE000, 0xF8FF)], "compat": [(0xF900, 0xFFFF)], "astral_scripts": [(0x10000, 0x10FFF)],
    "astral_more": [(0x11000, 0x16FFF)], "math_alnum": [(0x1D400, 0x1D7FF)],
    "emoji": [(0x1F300, 0x1FAFF), (0x2600, 0x27BF)], "emoji_mods": [(0x1F3FB, 0x1F3FF), (0x200D, 0x200D), (0xFE0F, 0xFE0F)],
    "flags": [(0x1F1E6, 0x1F1FF)], "cjk_b": [(0x20000, 0x2A6DF)], "tags_vs": [(0xE0000, 0xE01EF)],
    "plane15_16": [(0xF0000, 0x10FFFF)], "anything": [(0x0, 0xD7FF), (0xE000, 0x10FFFF)],
}

FUZZ_ALPHABETS = [
    "aAbB zZ",
    "aA'sStTmMdDrReEvVlL ",
    "aAéÉ\u0301 '",
    "ab AB\n\r\t /.,!",
    " \n\r\t\v\f\u00a0\u2003x",
    "019 a.,-/\n",
    "a1A \u0301\u0300é中.'s",
    "<|>_ abendmessagoftxu",
    "!?/\\\n\r a",
    "aZ0 .\n\u00e9\u0416\u4e2d\U0001F600\u0301\u00a0'",
    "𝐀𝐚Ａａǅʰªⅷ a",
    "\x00\x01\x1f\x7f\u0085\u200b\u200d\ufeff a.",
]


def rand_from(rng, ranges):
    lo, hi = ranges[rng.randrange(len(ranges))]
    return chr(rng.randint(lo, hi))


def random_string(rng, specials):
    kind = rng.random()
    n = int(rng.expovariate(1 / 24.0)) + 1
    if kind < 0.30:      # small-alphabet fuzz: exercises the splitter state machine
        alpha = rng.choice(FUZZ_ALPHABETS)
        return "".join(rng.choice(alpha) for _ in range(n))
    if kind < 0.55:      # one or two ranges
        names = rng.sample(sorted(RANGES), rng.choice([1, 1, 2, 2, 3]))
        return "".join(rand_from(rng, RANGES[rng.choice(names)]) for _ in range(n))
    if kind < 0.80:      # "words" from several ranges separated by assorted whitespace/punctuation
        names = rng.sample(sorted(RANGES), rng.randint(2, 6))
        out = []
        for _ in range(rng.randint(1, 12)):
            r = RANGES[rng.choice(names)]
            out.append("".join(rand_from(rng, r) for _ in range(rng.randint(1, 9))))
            out.append(rng.choice([" ", " ", " ", "", "\n", "  ", "\t", ", ", ". ", "'s ", "'", "/", "-", "\r\n", "\u00a0"]))
        return "".join(out)
    if kind < 0.92:      # anything at all
        return "".join(rand_from(rng, RANGES["anything"]) if rng.random() < 0.5
                       else rand_from(rng, RANGES[rng.choice(sorted(RANGES))]) for _ in range(n))
    # special tokens, fragments of them and text
    out = []
    for _ in range(rng.randint(1, 8)):
        r = rng.random()
        sp = rng.choice(specials)
        if r < 0.4:
            out.append(sp)
        elif r < 0.6:
            a = rng.randrange(len(sp))
            out.append(sp[a:rng.randint(a, len(sp))])
        elif r < 0.7:
            out.append(sp.upper() if rng.random() < 0.5 else sp.replace("|", "||", 1))
        else:
            out.append("".join(rng.choice("ab <|>_\n'sX1é") for _ in range(rng.randint(1, 6))))
    return "".join(out)


def random_bytes(rng):
    n = rng.randint(1, 40)
    kind = rng.random()
    if kind < 0.4:
        return bytes(rng.randrange(256) for _ in range(n))
    if kind < 0.7:       # valid text with a few corrupted bytes
        b = bytearray("".join(rand_from(rng, RANGES[rng.choice(sorted(RANGES))]) for _ in range(n)).encode("utf-8"))
        for _ in range(rng.randint(1, 3)):
            b[rng.randrange(len(b))] = rng.randrange(256)
        return bytes(b)
    b = "".join(rand_from(rng, RANGES["anything"]) for _ in range(n)).encode("utf-8")   # truncated
    return b[:rng.randint(1, len(b))]


def real_files(llama_src):
    """Real-world text: source code in several languages, docs, templates."""
    out = {}
    if not llama_src or not os.path.isdir(llama_src):
        return out
    pats = [
        ("README.md", 1), ("docs/*.md", 6), ("docs/backend/*.md", 2), ("src/llama-vocab.cpp", 1), ("src/unicode.cpp", 1),
        ("src/llama-sampler.cpp", 1), ("src/llama.cpp", 1), ("src/llama-model.h", 1), ("include/llama.h", 1),
        ("convert_hf_to_gguf.py", 1), ("gguf-py/gguf/constants.py", 1), ("gguf-py/gguf/vocab.py", 1),
        ("CMakeLists.txt", 1), ("src/CMakeLists.txt", 1), ("Makefile", 1), ("scripts/*.sh", 4), ("grammars/*.gbnf", 3),
        ("models/templates/*.jinja", 5), ("tools/server/README.md", 1), ("tools/server/server-common.cpp", 1),
        ("tools/server/webui/src/lib/**/*.ts", 4), ("tools/server/webui/src/lib/**/*.svelte", 3),
        ("tools/server/webui/package.json", 1), ("ggml/src/ggml-metal/*.metal", 1), ("ggml/src/ggml-metal/*.m", 1),
        ("examples/**/*.swift", 2), ("examples/**/*.kt", 2), ("examples/**/*.java", 1), (".github/workflows/*.yml", 2),
        ("common/*.cpp", 3), ("vendor/nlohmann/json.hpp", 1), ("pyproject.toml", 1), ("flake.nix", 1),
    ]
    for pat, limit in pats:
        paths = sorted(glob.glob(os.path.join(llama_src, pat), recursive=True))
        for p in paths[:limit]:
            try:
                data = open(p, "rb").read()
            except OSError:
                continue
            if data:
                out["file_" + os.path.relpath(p, llama_src).replace("/", "_")] = data[:1500000]
    # large inputs (200 KB+) of mixed code / prose
    cat = b"".join(open(p, "rb").read() for p in sorted(glob.glob(os.path.join(llama_src, "src", "llama-*.cpp")))[:12])
    if cat:
        out["big_cpp_400k"] = cat[:400000]
        out["big_cpp_tail_250k"] = cat[-250000:]
    md = b"".join(open(p, "rb").read() for p in sorted(glob.glob(os.path.join(llama_src, "**", "*.md"), recursive=True))[:400])
    if md:
        out["big_markdown_300k"] = md[:300000]
    return out


# ----------------------------------------------------------------------------------------------- running

def run_ref(ref_bin, ref_model, path, special):
    cmd = [ref_bin, "-m", ref_model, "-f", path, "--ids", "--no-escape", "--no-bos", "--log-disable"]
    if not special:
        cmd.append("--no-parse-special")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("reference failed on %s: %s" % (path, r.stderr.decode("utf-8", "replace")[-400:]))
    lines = [l for l in r.stdout.decode("ascii", "replace").splitlines() if l.startswith("[")]
    body = lines[-1].strip()[1:-1].strip()
    return [int(x) for x in body.split(",")] if body else []


def run_ours(tok_test, model, listfile, special):
    cmd = [tok_test, model, "batch", listfile]
    if not special:
        cmd.append("--no-special")
    r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if r.returncode != 0:
        raise RuntimeError("tok_test failed: " + r.stderr.decode("utf-8", "replace")[-400:])
    out = []
    for line in r.stdout.decode("ascii").splitlines():
        ids, rt = line.rsplit("|", 1)
        out.append(([int(x) for x in ids.split()], rt.strip() == "rt=1"))
    return out


def is_valid_utf8(b):
    try:
        b.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok-test", required=True)
    ap.add_argument("--ref-bin", required=True)
    ap.add_argument("--ref-model", required=True)
    ap.add_argument("--model", default=None, help="GGUF for our tokenizer (default: the reference model)")
    ap.add_argument("--llama-src", default=None, help="llama.cpp checkout used as a source of real text")
    ap.add_argument("--work", default=None)
    ap.add_argument("--random", type=int, default=4000, help="number of random strings")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--jobs", type=int, default=6)
    args = ap.parse_args()
    model = args.model or args.ref_model
    work = args.work or tempfile.mkdtemp(prefix="tok_compare_")
    os.makedirs(work, exist_ok=True)
    rng = random.Random(args.seed)

    r = subprocess.run([args.tok_test, model, "specials"], stdout=subprocess.PIPE, check=True)
    special_ids, specials = {}, []
    for line in r.stdout.decode("utf-8").splitlines():
        i, t = line.split("\t", 1)
        special_ids[t] = int(i)
        specials.append(t)
    for sp in SPECIALS_USED:
        if sp not in special_ids:
            sys.exit("expected special token %s not in vocabulary" % sp)

    # name -> (bytes, n_strings).  Random strings are batched: many strings joined by a special token, which the
    # tokenizers treat as a hard boundary (each string is tokenized independently), so one reference process
    # checks a few hundred strings.  The joined text is also compared without special parsing.
    samples = {}
    for k, v in handwritten(specials).items():
        samples[k] = (v, 1)
    for k, v in invalid_utf8().items():
        samples[k] = (v, 1)
    for k, v in real_files(args.llama_src).items():
        samples[k] = (v, 1)

    sep = "<|endoftext|>"
    per_batch = 250
    n_random = 0
    for b in range((args.random + per_batch - 1) // per_batch):
        strs = [random_string(rng, specials) for _ in range(per_batch)]
        n_random += len(strs)
        samples["random_batch_%03d" % b] = (sep.join(strs).encode("utf-8"), len(strs))
    n_random_bytes = 0
    for b in range(max(1, args.random // 2000)):
        strs = [random_bytes(rng) for _ in range(per_batch)]
        n_random_bytes += len(strs)
        samples["random_bytes_batch_%03d" % b] = (sep.encode().join(strs), len(strs))
    # a few random strings on their own (no batching), including ones starting/ending in odd places
    for i in range(40):
        samples["random_single_%03d" % i] = (random_string(rng, specials).encode("utf-8"), 1)

    names = sorted(samples)
    paths = {}
    for i, name in enumerate(names):
        p = os.path.join(work, "s%04d.txt" % i)
        with open(p, "wb") as f:
            f.write(samples[name][0])
        paths[name] = p
    listfile = os.path.join(work, "list.txt")
    with open(listfile, "w") as f:
        f.write("\n".join(paths[n] for n in names) + "\n")

    ours = {True: run_ours(args.tok_test, model, listfile, True), False: run_ours(args.tok_test, model, listfile, False)}
    for mode in (True, False):
        if len(ours[mode]) != len(names):
            sys.exit("tok_test returned %d results for %d samples" % (len(ours[mode]), len(names)))

    jobs = [(n, mode) for n in names for mode in (True, False)]
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        refs = list(ex.map(lambda j: run_ref(args.ref_bin, args.ref_model, paths[j[0]], j[1]), jobs))

    mismatches = 0
    rt_fail = 0
    rt_checked = 0
    total_tokens = 0
    total_bytes = 0
    comparisons = 0
    strings = 0
    for (name, mode), ref in zip(jobs, refs):
        mine, rt = ours[mode][names.index(name)]
        data, count = samples[name]
        comparisons += 1
        strings += count
        total_tokens += len(ref)
        total_bytes += len(data)
        if mine != ref:
            mismatches += 1
            k = next((i for i in range(min(len(mine), len(ref))) if mine[i] != ref[i]), min(len(mine), len(ref)))
            print("MISMATCH %s (parse_special=%s): %d vs %d tokens, first difference at %d: ours %s ref %s   file %s"
                  % (name, mode, len(mine), len(ref), k, mine[k:k + 6], ref[k:k + 6], paths[name]))
        if is_valid_utf8(data):
            rt_checked += 1
            if not rt:
                rt_fail += 1
                print("ROUNDTRIP FAIL %s (parse_special=%s)   file %s" % (name, mode, paths[name]))

    print("samples: %d files (%d hand-written/real, %d random strings in batches, %d random byte strings), "
          "each compared with and without special parsing"
          % (len(names), len(names) - sum(1 for n in names if n.startswith("random_")) , n_random + 40, n_random_bytes))
    print("comparisons: %d (covering %d strings), bytes: %d, tokens compared: %d" % (comparisons, strings, total_bytes, total_tokens))
    print("mismatches: %d" % mismatches)
    print("round-trip decode(encode(x)) == x: %d checked, %d failed" % (rt_checked, rt_fail))
    print("work dir: %s" % work)
    sys.exit(1 if (mismatches or rt_fail) else 0)


if __name__ == "__main__":
    main()
