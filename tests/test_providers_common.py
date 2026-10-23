import json

from legal_analyzer.citations import cited_passages, parse_inline_citations
from legal_analyzer.llm import ChatTurn
from legal_analyzer.models import Chunk
from legal_analyzer.providers.common import ReasoningFilter, chat_messages, extract_json, strict_json_schema
from legal_analyzer.schemas import PassOutput, SynthesisOutput


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from _walk(item)


def test_strict_schema_is_self_contained_and_closed():
    for model in (PassOutput, SynthesisOutput):
        schema = strict_json_schema(model)
        text = json.dumps(schema)
        assert "$ref" not in text and "$defs" not in text
        for node in _walk(schema):
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
    finding = strict_json_schema(PassOutput)["properties"]["findings"]["items"]
    assert "title" in finding["properties"]  # a property named "title" survives keyword stripping
    assert "limitation_of_liability" in finding["properties"]["category_id"]["enum"]


def test_parse_inline_citations():
    text = "Fees are non-refundable [C4]. They can end it anytime [C9, C2][C9]. Fake [C99]. Footnote [1]."
    segments = parse_inline_citations(text, {2, 4, 9})
    assert segments == [
        ("Fees are non-refundable", [4]),
        (". They can end it anytime", [9, 2]),
        (". Fake", []),
        (". Footnote [1].", []),
    ]


def test_reasoning_filter_hides_split_think_tags():
    f = ReasoningFilter()
    pieces = ["<thi", "nk>secret plan</th", "ink>Answer: ", "fees are <", "b>bold</b> fine"]
    shown = "".join(f.feed(p) for p in pieces) + f.flush()
    assert shown == "Answer: fees are <b>bold</b> fine"


def test_extract_json_strips_fences_and_reasoning():
    assert extract_json('<think>hmm</think>\n```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json('Here you go: {"a": {"b": 2}} done') == '{"a": {"b": 2}}'


def _chunk(i, text):
    return Chunk(id=i, text=text, page_start=1, page_end=1, section=f"{i}. SECTION")


def test_chat_messages_keep_recent_history_within_budget():
    old = ChatTurn("first question " * 50, [], [], reply="first answer " * 50)
    recent = ChatTurn("second question", [], [], reply="second answer [C1]")
    current = ChatTurn("third question", [_chunk(1, "Fees are non-refundable.")], [1])
    messages = chat_messages("SYSTEM", [old, recent, current], history_chars=200)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[1]["content"] == "second question"
    assert '<excerpt id="C1"' in messages[-1]["content"] and messages[-1]["content"].endswith("Question: third question")


def test_cited_passages_pick_the_supporting_sentence():
    chunk = _chunk(3, "3.1 You must be 18 or older.\n3.2 Acme may refuse service to anyone for any reason at any time.")
    segments = [("Acme can refuse service to anyone for any reason", [3])]
    assert cited_passages(segments, {3: chunk}) == {3: ["3.2 Acme may refuse service to anyone for any reason at any time."]}
