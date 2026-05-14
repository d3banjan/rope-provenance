"""Tag → role span parser.

Standalone module: no torch / transformers dependency. The character-level
parser is the load-bearing security boundary — the model is trained to
respect whatever role_ids this produces, so adversarial markup inside an
untrusted span must not be able to escalate trust.

Architectural premise: tag scanning is *stateful*. Once inside a span of
rule R, only R's own close tag (and, depending on policy, R's own nested
open) is recognized as markup; tags belonging to other rules are content.
This is what makes the case "DATA span containing literal `<|instruction|>`"
safe — that lookalike never re-enters the parser's open-tag scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import yaml


class MalformedMarkupError(ValueError):
    """Unclosed or syntactically broken markup."""


class OverlapError(ValueError):
    """Nested same-rule open under ``overlap_policy=error``."""


OverlapPolicy = Literal["error", "innermost-wins", "outermost-wins"]
UnclosedPolicy = Literal["error", "treat-as-default"]


@dataclass(frozen=True)
class SpanRule:
    open: str
    close: str
    role: str  # role name; resolved to id via RoleMap.roles


@dataclass(frozen=True)
class RoleMap:
    roles: dict[str, int]
    default_role: str
    spans: tuple[SpanRule, ...]
    overlap_policy: OverlapPolicy = "error"
    on_unclosed: UnclosedPolicy = "error"

    @property
    def default_id(self) -> int:
        return self.roles[self.default_role]

    def role_id(self, name: str) -> int:
        return self.roles[name]

    @classmethod
    def from_dict(cls, d: dict) -> "RoleMap":
        roles = {str(k): int(v) for k, v in d["roles"].items()}
        spans = tuple(
            SpanRule(open=s["open"], close=s["close"], role=s["role"])
            for s in d.get("spans", [])
        )
        for s in spans:
            if s.role not in roles:
                raise ValueError(
                    f"Span rule references unknown role {s.role!r}; known: {list(roles)}"
                )
        default = d.get("default_role")
        if default is None:
            raise ValueError("role_map missing 'default_role'")
        if default not in roles:
            raise ValueError(f"default_role {default!r} not in roles {list(roles)}")
        return cls(
            roles=roles,
            default_role=default,
            spans=spans,
            overlap_policy=d.get("overlap_policy", "error"),
            on_unclosed=d.get("on_unclosed", "error"),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RoleMap":
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))


def _find_earliest(
    text: str, start: int, needles: Sequence[tuple[str, object]]
) -> tuple[int, str, object] | None:
    """Return earliest occurrence of any needle in `text[start:]`.

    On ties (same start position), longer needle wins — so a rule open
    that is a strict prefix of another is matched as the longer one.
    """
    best: tuple[int, str, object] | None = None
    for needle, tag in needles:
        if not needle:
            continue
        idx = text.find(needle, start)
        if idx == -1:
            continue
        if best is None or idx < best[0] or (idx == best[0] and len(needle) > len(best[1])):
            best = (idx, needle, tag)
    return best


def parse_char_roles(text: str, role_map: RoleMap) -> list[int]:
    """Return per-character role id array of length ``len(text)``.

    Stateful scan: from outside any span, recognize every rule's open tag;
    inside rule R's span, recognize only R's own close tag (always) and
    R's own open tag (for nested-same-rule under overlap policies).
    Foreign-rule tags inside an open span are content.
    """
    n = len(text)
    out = [role_map.default_id] * n
    pos = 0

    while pos < n:
        outside_candidates: list[tuple[str, object]] = [
            (r.open, ("open", r)) for r in role_map.spans
        ]
        hit = _find_earliest(text, pos, outside_candidates)
        if hit is None:
            break
        open_idx, open_tag, (_kind, rule) = hit  # type: ignore[misc]
        assert isinstance(rule, SpanRule)
        # Content from `pos` up to `open_idx` keeps default_id (already set).
        content_start = open_idx + len(open_tag)
        depth = 1
        cursor = content_start
        role_id = role_map.role_id(rule.role)

        while depth > 0:
            inner_needles: list[tuple[str, object]] = [(rule.close, "close")]
            # outermost-wins: do not recognize nested same-rule opens at all;
            # the first close ends the outermost span.
            if role_map.overlap_policy != "outermost-wins":
                inner_needles.append((rule.open, "nested-open"))
            inner_hit = _find_earliest(text, cursor, inner_needles)
            if inner_hit is None:
                if role_map.on_unclosed == "error":
                    raise MalformedMarkupError(
                        f"Unclosed span: rule {rule.role!r} opened at offset "
                        f"{open_idx} (tag {rule.open!r}); close tag {rule.close!r} "
                        f"not found"
                    )
                # treat-as-default: characters from content_start onward keep
                # default_id; out is already initialized that way.
                return out
            ev_idx, ev_tag, kind = inner_hit
            if kind == "close":
                # Paint the content range. Even with nested opens stacked
                # via innermost-wins, the role here is the same as outer,
                # so painting on each pop is idempotent.
                for i in range(content_start, ev_idx):
                    out[i] = role_id
                depth -= 1
                if depth > 0:
                    # innermost-wins: continue scanning for next close in the
                    # remaining open levels; content_start unchanged so the
                    # outermost range is painted on the final close.
                    cursor = ev_idx + len(ev_tag)
                    continue
                # Hand off to outside-scan starting AT the close-tag position
                # (not past it). This lets a tag that serves dual duty — close
                # of rule X and open of rule Y — immediately re-enter rule Y.
                # Example: in the chat template `<|instruction|>{i}<|data|>{d}<|assistant|>`,
                # `<|data|>` is both INSTRUCTION's close and DATA's open.
                pos = ev_idx
            else:  # nested-open
                if role_map.overlap_policy == "error":
                    raise OverlapError(
                        f"Nested {rule.open!r} at offset {ev_idx}; previous open "
                        f"at {open_idx}. overlap_policy=error forbids nesting."
                    )
                # innermost-wins: deepen the stack and continue searching.
                depth += 1
                cursor = ev_idx + len(ev_tag)
    return out


def _majority_role(roles: Iterable[int], default: int) -> int:
    counts: dict[int, int] = {}
    for r in roles:
        counts[r] = counts.get(r, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda kv: kv[1])[0]


def parse_to_role_ids(
    text: str,
    tokenizer,
    role_map: RoleMap,
) -> tuple[list[int], list[int]]:
    """Tokenize ``text`` and produce aligned (input_ids, role_ids).

    The tokenizer is duck-typed: it must accept
    ``tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)``
    and return an object with ``input_ids`` and ``offset_mapping`` attributes
    or keys. Both fast HF tokenizers and a stub for testing satisfy this.
    """
    char_roles = parse_char_roles(text, role_map)
    enc = tokenizer(
        text,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"] if isinstance(enc, dict) else enc.input_ids
    offsets = enc["offset_mapping"] if isinstance(enc, dict) else enc.offset_mapping
    role_ids: list[int] = []
    for start, end in offsets:
        if start == end:  # special token, no chars
            role_ids.append(role_map.default_id)
            continue
        span_roles = char_roles[start:end]
        first = span_roles[0]
        if any(r != first for r in span_roles):
            # Token straddles a boundary — majority wins, log via WARNING.
            import logging
            logging.getLogger(__name__).warning(
                "Token at offsets [%d, %d) straddles role boundary: %s",
                start, end, span_roles,
            )
            role_ids.append(_majority_role(span_roles, role_map.default_id))
        else:
            role_ids.append(first)
    return list(input_ids), role_ids
