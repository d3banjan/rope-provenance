"""Tests for the prompt-rendering + role-tagging collator.

Uses the SmolLM2 tokenizer (small, cached after the first download). The
collator is the only place where parser char-level roles meet real tokens,
so the alignment between role boundaries and tokenizer offsets is checked
exhaustively here, not in test_parser_fuzz.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rope_prov.data import (
    ASSISTANT_MARKER,
    PROMPT_TEMPLATE,
    RoleTaggingCollator,
    build_counterfactual_examples,
    filter_alpaca_examples,
    render_example,
)
from rope_prov.parser import RoleMap


ROLE_MAP_PATH = (
    Path(__file__).parent.parent / "src" / "rope_prov" / "configs" / "role_map.yaml"
)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


@pytest.fixture(scope="module")
def role_map():
    return RoleMap.from_yaml(ROLE_MAP_PATH)


@pytest.fixture(scope="module")
def collator(tokenizer, role_map):
    return RoleTaggingCollator(
        tokenizer=tokenizer, role_map=role_map, max_length=512, pad_to_multiple_of=8
    )


_EXAMPLES = [
    {
        "instruction": "Summarize the email below.",
        "input": "Hi team, please review the attached doc by EOD.",
        "output": "The email asks the team to review an attachment by end of day.",
    },
    {
        "instruction": "Translate the text to French.",
        "input": "Hello, how are you today?",
        "output": "Bonjour, comment allez-vous aujourd'hui?",
    },
    {
        "instruction": "List the entities in the text.",
        "input": "Alice met Bob in Paris in 2021.",
        "output": "Alice, Bob, Paris, 2021.",
    },
]


# ---------- rendering -------------------------------------------------------


def test_render_example_marker_offset_is_correct():
    text, asst_start = render_example(_EXAMPLES[0])
    assert text[asst_start - len(ASSISTANT_MARKER) : asst_start] == ASSISTANT_MARKER
    assert text.startswith("<|system|>")
    assert text[asst_start:] == _EXAMPLES[0]["output"]


def test_filter_alpaca_drops_empty_input():
    raw = [
        {"instruction": "x", "input": "", "output": "y"},
        {"instruction": "x", "input": "z", "output": "y"},
        {"instruction": "x", "input": "z", "output": ""},
    ]
    kept = filter_alpaca_examples(raw)
    assert len(kept) == 1
    assert kept[0]["input"] == "z"


# ---------- collator output -------------------------------------------------


def test_collator_emits_required_keys_and_shapes(collator):
    batch = collator(_EXAMPLES)
    assert set(batch) == {"input_ids", "attention_mask", "labels", "role_ids"}
    B, T = batch["input_ids"].shape
    assert B == len(_EXAMPLES)
    assert batch["role_ids"].shape == (B, T)
    assert batch["labels"].shape == (B, T)
    assert batch["attention_mask"].shape == (B, T)
    # pad_to_multiple_of = 8
    assert T % 8 == 0


def test_every_token_has_valid_role(collator, role_map):
    """Hermetic coverage: no token ends up unlabeled."""
    batch = collator(_EXAMPLES)
    valid = set(role_map.roles.values())
    flat = batch["role_ids"].flatten().tolist()
    assert set(flat).issubset(valid), (
        f"role_ids contain values outside {valid}: {set(flat) - valid}"
    )


def test_first_batch_has_more_than_one_unique_role(collator):
    """Step-0 collator sanity: if this fails, the parser silently emitted
    all-default-role tokens and rope-prov training will regress to vanilla."""
    batch = collator(_EXAMPLES)
    unique = batch["role_ids"].unique().tolist()
    assert len(unique) > 1, (
        f"role_ids has only one unique value {unique}; collator regression "
        f"would silently train rope-prov as vanilla."
    )


def test_labels_mask_everything_before_assistant_content(collator, tokenizer):
    """Loss should apply only to assistant tokens. Everything else = -100."""
    batch = collator([_EXAMPLES[0]])
    labels = batch["labels"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()
    attn = batch["attention_mask"][0].tolist()

    # First non-(-100) label should be the first assistant token.
    assistant_label_positions = [i for i, l in enumerate(labels) if l != -100]
    assert assistant_label_positions, "no assistant tokens have a loss target"
    first_assistant_pos = assistant_label_positions[0]

    # Labels at assistant positions equal the corresponding input_ids
    # (standard causal LM target).
    for i in assistant_label_positions:
        assert labels[i] == input_ids[i]
        assert attn[i] == 1

    # Pre-assistant content (the system prompt, instruction, data sections)
    # must be entirely masked.
    for i in range(first_assistant_pos):
        assert labels[i] == -100


def test_data_section_tokens_are_data_role(collator, role_map, tokenizer):
    """The contiguous DATA span in the rendered prompt must be tagged DATA
    end-to-end. Pick the example whose input string is recognizable so we
    can locate it post-tokenization."""
    ex = _EXAMPLES[0]
    batch = collator([ex])
    role_ids = batch["role_ids"][0].tolist()
    input_ids = batch["input_ids"][0].tolist()
    text = tokenizer.decode(input_ids, skip_special_tokens=False)

    # The data string should appear verbatim in the decoded text — find its
    # token-index range via offsets recomputed deterministically here.
    enc = tokenizer(
        render_example(ex)[0],
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=512,
    )
    offsets = enc["offset_mapping"]
    full_text = render_example(ex)[0]
    data_str = ex["input"]
    data_char_start = full_text.index(data_str)
    data_char_end = data_char_start + len(data_str)

    DATA_ID = role_map.roles["DATA"]
    for i, (s, e) in enumerate(offsets):
        if s >= data_char_start and e <= data_char_end:
            assert role_ids[i] == DATA_ID, (
                f"token at offset [{s},{e}) inside DATA span got role "
                f"{role_ids[i]}, expected {DATA_ID}"
            )


def test_idempotence(collator):
    """parse(text) == parse(text) byte-for-byte."""
    b1 = collator(_EXAMPLES)
    b2 = collator(_EXAMPLES)
    for k in b1:
        assert torch.equal(b1[k], b2[k]), f"key {k!r} not idempotent"


def test_counterfactual_builder_creates_matched_roles(collator, role_map):
    examples = build_counterfactual_examples(12, seed=123, positive_fraction=0.5)
    assert len(examples) == 12
    assert {
        "synthetic_counterfactual_data_negative",
        "synthetic_counterfactual_instruction_positive",
    }.issubset({ex["source"] for ex in examples})

    # Both classes must produce loss-bearing assistant tokens and both role ids.
    batch = collator(examples)
    assert (batch["labels"] != -100).any()
    assert set(batch["role_ids"].unique().tolist()) == set(role_map.roles.values())


def test_data_negative_counterfactual_labels_ignore_injected_directive():
    examples = build_counterfactual_examples(20, seed=0, positive_fraction=0.0)
    for ex in examples:
        assert ex["source"] == "synthetic_counterfactual_data_negative"
        assert ex["counterfactual_witness"] in ex["input"]
        assert ex["output"] != ex["counterfactual_witness"]
        assert ex["counterfactual_witness"] not in ex["instruction"]


def test_instruction_positive_counterfactual_labels_follow_same_directive():
    examples = build_counterfactual_examples(20, seed=0, positive_fraction=1.0)
    for ex in examples:
        assert ex["source"] == "synthetic_counterfactual_instruction_positive"
        assert ex["output"] in ex["instruction"]
        assert ex["output"] not in ex["input"]


def test_counterfactual_builder_covers_diversity_axes():
    examples = build_counterfactual_examples(120, seed=0, positive_fraction=0.5)
    assert {ex["counterfactual_task_type"] for ex in examples} == {
        "field_extraction",
        "factual_lookup",
        "computation",
        "generation",
        "refusal_label",
    }
    assert {ex["counterfactual_directive_style"] for ex in examples} == {
        "imperative",
        "question",
        "assertion_then_action",
        "embedded",
        "systemish",
        "json",
    }
    assert {ex["counterfactual_placement"] for ex in examples} == {
        "early",
        "middle",
        "late",
        "multiple",
    }
    assert {ex["counterfactual_distractor_difficulty"] for ex in examples} == {
        "mundane",
        "semantic_fit",
    }


def test_counterfactual_pairs_reuse_same_directive_across_roles():
    examples = build_counterfactual_examples(40, seed=0, positive_fraction=0.5)
    by_pair: dict[int, list[dict]] = {}
    for ex in examples:
        by_pair.setdefault(ex["counterfactual_pair_id"], []).append(ex)

    assert by_pair
    for pair in by_pair.values():
        assert {ex["source"] for ex in pair} == {
            "synthetic_counterfactual_data_negative",
            "synthetic_counterfactual_instruction_positive",
        }
        negative = next(
            ex
            for ex in pair
            if ex["source"] == "synthetic_counterfactual_data_negative"
        )
        positive = next(
            ex
            for ex in pair
            if ex["source"] == "synthetic_counterfactual_instruction_positive"
        )
        witness = positive["counterfactual_witness"]
        assert positive["output"] == witness
        assert witness in positive["instruction"]
        assert witness in negative["input"]
