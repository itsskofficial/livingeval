"""The canonical trace-to-text rendering.

There is exactly one implementation of this in the library, and every downstream
number depends on it: clustering, coverage, the scorer ladder and any embedding
backend all read text produced here.

That single-source rule is the point. A common way to get an inflated agreement
number out of a distilled scorer is to feed the scorer a rendering that includes
something the judge could not see, or that leaks the label - a `status: error` field
in `meta`, a `verdict` key someone left on a turn, the tool's own success flag.
`render()` reads `role`, `name` and `content` and nothing else, so a leak has to be
introduced deliberately rather than by accident.

Four views ship:

- `render` - the whole conversation. The default for judging and for the ladder,
  because it is what a judge is usually shown.
- `render_request` - the user turns only. The default for clustering and coverage,
  because a blind spot is a region of *demand*: it is a thing your users are asking
  for that your suite has no case for. Cluster the full trace instead and a large
  share of the partition ends up describing your own response templates - "the
  cluster where the agent says it has logged the reference" - which is a fact about
  your prompt, not about your traffic, and which moves every time you edit that
  prompt.
- `render_response` - the final assistant turn only, which is what a
  response-quality judge is actually shown. Scoring against the whole trace when the
  judge saw one turn makes the ladder answer a different question than the one asked.
- `render_last_exchange` - the last user/assistant pair, the usual middle ground.
"""

from __future__ import annotations

from livingeval.trace.types import Trace

__all__ = [
    "VIEWS",
    "render",
    "render_last_exchange",
    "render_request",
    "render_response",
    "render_view",
]

_PREFIX = {"system": "system", "user": "user", "assistant": "assistant", "tool": "tool"}


def _line(role: str, name: str | None, content: str) -> str:
    tag = _PREFIX[role] if name is None else f"{_PREFIX[role]}:{name}"
    return f"<{tag}> {content.strip()}"


def render(trace: Trace, include_system: bool = False) -> str:
    """Full conversation. System turns are excluded by default because they are
    usually constant across a corpus and add a large shared prefix that dominates
    tf-idf similarity."""
    parts = [
        _line(t.role, t.name, t.content)
        for t in trace.turns
        if include_system or t.role != "system"
    ]
    return "\n".join(parts)


def render_request(trace: Trace) -> str:
    """The user turns only - what was asked, with nothing the agent said."""
    return "\n".join(t.content.strip() for t in trace.turns if t.role == "user")


def render_response(trace: Trace) -> str:
    """The final assistant turn only."""
    last = trace.last("assistant")
    return "" if last is None else last.content.strip()


def render_last_exchange(trace: Trace) -> str:
    """The last user turn, any tool turns after it, and the final assistant turn."""
    turns = trace.turns
    start = 0
    for i in range(len(turns) - 1, -1, -1):
        if turns[i].role == "user":
            start = i
            break
    return "\n".join(_line(t.role, t.name, t.content) for t in turns[start:] if t.role != "system")


VIEWS = {
    "full": render,
    "request": render_request,
    "response": render_response,
    "last_exchange": render_last_exchange,
}


def render_view(trace: Trace, view: str = "full") -> str:
    """Dispatch by name. The view name is recorded in every result, because two
    coverage numbers computed under different views are not comparable."""
    try:
        fn = VIEWS[view]
    except KeyError:
        raise ValueError(f"unknown view {view!r}; choose from {sorted(VIEWS)}") from None
    return fn(trace)
