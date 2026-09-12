"""Finding the LLM call sites in a repository.

Static analysis, deliberately. Importing a codebase to inspect it means running
its import side effects -- connecting to databases, reading credentials,
occasionally spending money -- on a machine whose owner asked for a scan, not an
execution. `ast` sees enough.

What "enough" means in practice: the provider call itself is easy, and the
interesting part is the evidence *around* it. A call to `chat.completions.create`
tells you almost nothing. The same call in a function that also invoked a
retriever, with a `response_format` naming a Pydantic model, inside a loop over
`tool_calls`, tells you what to evaluate. So the visitor collects signals from
the enclosing function rather than the call expression alone.

Reported confidence is about *archetype*, never about whether a call site exists
-- a matched provider call is a fact. Sites the classifier cannot place come back
as `GENERIC`, which still earns the universal metrics. Silence is not an option:
an LLM call nobody evaluates is exactly what this tool exists to find.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path

__all__ = ["CallSite", "Evidence", "scan_file", "scan_tree"]

# Attribute chains that mean "a model was called". Matched on the tail of the
# dotted name, so `client.chat.completions.create` and
# `self._openai.chat.completions.create` both hit.
PROVIDER_CALLS: dict[tuple[str, ...], str] = {
    ("chat", "completions", "create"): "openai",
    ("completions", "create"): "openai",
    ("responses", "create"): "openai",
    ("messages", "create"): "anthropic",
    ("messages", "stream"): "anthropic",
    ("generate_content",): "google",
}

# Chains too generic to name a provider on their own. `client.chat(...)` is
# Cohere, Ollama or somebody's own wrapper depending only on what the file
# imported, so these resolve against the module's imports and are ignored when
# nothing matches -- guessing here produces confidently mislabelled sites.
AMBIGUOUS_CALLS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("chat",): ("cohere", "ollama", "mistralai", "groq"),
    ("generate",): ("ollama", "cohere"),
}

# Bare function calls with the same meaning.
PROVIDER_FUNCTIONS: dict[str, str] = {
    "completion": "litellm",
    "acompletion": "litellm",
}

# LangChain and LlamaIndex hide the provider behind a runnable, so the call is
# `.invoke` on something that was constructed from a chat model. The visitor
# tracks those constructions per module and treats invocations of the resulting
# names as call sites.
CHAT_MODEL_CTORS = {
    "ChatOpenAI", "ChatAnthropic", "ChatGoogleGenerativeAI", "ChatGroq",
    "ChatOllama", "ChatCohere", "ChatMistralAI", "ChatBedrock", "ChatVertexAI",
    "AzureChatOpenAI", "ChatLiteLLM", "OpenAI", "Anthropic", "LlamaCPP",
    "HuggingFaceEndpoint", "ChatHuggingFace",
}

# Things that are not chat models but are invoked exactly like one: a compiled
# graph, an agent executor. Tracked alongside the chat models because the call
# site is `agent.ainvoke(...)` either way, and in an agent codebase that is
# where every model call lives.
RUNNABLE_CTORS = {
    "StateGraph", "MessageGraph", "Graph", "AgentExecutor",
    "create_react_agent", "create_tool_calling_agent", "initialize_agent",
    "create_openai_functions_agent", "create_structured_chat_agent",
    "LLMChain", "ConversationChain", "RetrievalQA",
}

INVOKE_METHODS = {"invoke", "ainvoke", "stream", "astream", "batch", "abatch",
                  "predict", "run", "complete", "acomplete"}

RETRIEVAL_MARKERS = {
    "similarity_search", "similarity_search_with_score", "as_retriever",
    "max_marginal_relevance_search", "get_relevant_documents", "retrieve",
    "aretrieve", "query", "search", "vector_search", "hybrid_search",
}

RERANK_MARKERS = {"rerank", "compress_documents", "ContextualCompressionRetriever"}

STRUCTURED_MARKERS = {"with_structured_output", "parse", "model_validate",
                      "PydanticOutputParser", "JsonOutputParser"}

TOOL_MARKERS = {"bind_tools", "AgentExecutor", "create_react_agent",
                "create_tool_calling_agent", "initialize_agent", "ToolNode",
                "create_openai_functions_agent", "FunctionTool"}

MEMORY_MARKERS = {"ConversationBufferMemory", "ConversationSummaryMemory",
                  "ChatMessageHistory", "RunnableWithMessageHistory",
                  "ConversationBufferWindowMemory"}

SUMMARY_WORDS = ("summarize", "summarise", "summary", "tl;dr", "condense",
                 "abstract of", "key points")


@dataclass
class Evidence:
    """What the enclosing function does, besides call a model.

    Archetype classification reads this rather than the call expression, because
    the call expression is the same shape whatever the application does.
    """

    retrieval: list[str] = field(default_factory=list)
    rerank: list[str] = field(default_factory=list)
    structured: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    memory: list[str] = field(default_factory=list)
    schemas: list[str] = field(default_factory=list)
    literal_enums: list[str] = field(default_factory=list)
    # schema name -> (field count, whether every field is a label set)
    schema_shapes: dict[str, tuple[int, bool]] = field(default_factory=dict)
    # schema name -> [{"name", "required", "labels"}]. The counts above decide
    # the archetype; these decide whether the structured-output metrics can run
    # at all, because "required_fields" cannot be checked without the names and
    # the visitor is already standing in front of them.
    schema_fields: dict[str, list[dict]] = field(default_factory=dict)
    prompts: list[str] = field(default_factory=list)
    has_loop: bool = False
    streams: bool = False

    def merge(self, other: Evidence) -> None:
        self.retrieval += other.retrieval
        self.rerank += other.rerank
        self.structured += other.structured
        self.tools += other.tools
        self.memory += other.memory
        self.schemas += other.schemas
        self.literal_enums += other.literal_enums
        self.schema_shapes.update(other.schema_shapes)
        self.schema_fields.update(other.schema_fields)
        self.prompts += other.prompts
        self.has_loop = self.has_loop or other.has_loop
        self.streams = self.streams or other.streams


@dataclass
class CallSite:
    """One place a model is called, and everything known about it."""

    path: Path
    line: int
    function: str
    provider: str
    model: str | None = None
    evidence: Evidence = field(default_factory=Evidence)
    archetype: str = "generic"
    confidence: float = 0.0
    rationale: tuple[str, ...] = ()

    @property
    def ident(self) -> str:
        """Stable id, used to name generated files and to match across runs."""
        return f"{self.path.stem}.{self.function}:{self.line}"


def _dotted(node: ast.AST) -> tuple[str, ...]:
    """The attribute chain of a call target, outermost last."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return tuple(reversed(parts))


def _literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _kwarg(call: ast.Call, name: str) -> ast.AST | None:
    return next((k.value for k in call.keywords if k.arg == name), None)


class _FunctionVisitor(ast.NodeVisitor):
    """Walks one function body, gathering evidence and provider calls."""

    def __init__(self, chat_models: set[str], imports: set[str]) -> None:
        self.chat_models = chat_models
        self.imports = imports
        self.evidence = Evidence()
        self.calls: list[tuple[int, str, str | None]] = []   # line, provider, model

    # -- evidence ---------------------------------------------------------
    def visit_For(self, node: ast.For) -> None:
        self.evidence.has_loop = True
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.evidence.has_loop = True
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        # Long string literals in a function that calls a model are almost
        # always the prompt. Kept for archetype hints and to seed the rubric.
        if isinstance(node.value, str) and len(node.value) > 40:
            self.evidence.prompts.append(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        chain = _dotted(node.func)
        tail = chain[-1] if chain else ""

        for marker_set, bucket in (
            (RETRIEVAL_MARKERS, self.evidence.retrieval),
            (RERANK_MARKERS, self.evidence.rerank),
            (STRUCTURED_MARKERS, self.evidence.structured),
            (TOOL_MARKERS, self.evidence.tools),
            (MEMORY_MARKERS, self.evidence.memory),
        ):
            if tail in marker_set:
                bucket.append(".".join(chain))

        provider = self._provider_for(chain, tail)
        if provider:
            self.calls.append((node.lineno, provider, self._model_for(node, chain)))
            self._structured_kwargs(node)
            if tail in {"stream", "astream"} or _kwarg(node, "stream"):
                self.evidence.streams = True

        self.generic_visit(node)

    # -- provider detection ------------------------------------------------
    def _provider_for(self, chain: tuple[str, ...], tail: str) -> str | None:
        for suffix, provider in PROVIDER_CALLS.items():
            if len(chain) >= len(suffix) and chain[-len(suffix):] == suffix:
                return provider
        if len(chain) == 1 and tail in PROVIDER_FUNCTIONS:
            return PROVIDER_FUNCTIONS[tail]
        # Generic method names, resolved by what the file imported.
        for suffix, candidates in AMBIGUOUS_CALLS.items():
            if len(chain) >= len(suffix) + 1 and chain[-len(suffix):] == suffix:
                match = next((c for c in candidates if c in self.imports), None)
                if match:
                    return match
        # A runnable built from a chat model, invoked. Any segment before the
        # method may be the tracked name, not just the first: `self.llm` and
        # `self._deps.chat` are how these objects are actually held, and
        # matching only chain[0] finds `llm.invoke()` in a script while missing
        # every call in a class.
        if tail in INVOKE_METHODS and len(chain) >= 2 and any(
                part in self.chat_models for part in chain[:-1]):
            return "langchain"
        return None

    def _model_for(self, node: ast.Call, chain: tuple[str, ...]) -> str | None:
        for key in ("model", "model_name", "deployment_name"):
            if (value := _literal(_kwarg(node, key))) is not None:
                return value
        return None

    def _structured_kwargs(self, node: ast.Call) -> None:
        """Schema evidence carried on the call itself."""
        fmt = _kwarg(node, "response_format")
        if fmt is not None:
            self.evidence.structured.append("response_format")
            if isinstance(fmt, ast.Name):
                self.evidence.schemas.append(fmt.id)
            elif isinstance(fmt, ast.Attribute):
                self.evidence.schemas.append(fmt.attr)
        if _kwarg(node, "tools") is not None or _kwarg(node, "functions") is not None:
            self.evidence.tools.append("tools=")
        if (schema := _kwarg(node, "text_format")) is not None:
            self.evidence.structured.append("text_format")
            if isinstance(schema, ast.Name):
                self.evidence.schemas.append(schema.id)


class _ModuleVisitor(ast.NodeVisitor):
    """Finds chat-model constructions, then walks each function separately."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.imports: set[str] = set()
        self.chat_models: set[str] = set()
        self.enums: list[str] = []
        self.schema_shapes: dict[str, tuple[int, bool]] = {}
        self.schema_fields: dict[str, list[dict]] = {}
        self.sites: list[CallSite] = []

    def collect_bindings(self, tree: ast.AST) -> None:
        """Run to a fixed point.

        One pass is source order, and source order is not definition order: a
        method that builds a graph is usually written below the method that
        calls it, so `chain = workflow.compile()` is seen before anything knows
        what `workflow` is. Repeating until nothing new is learned costs two or
        three walks of a file and finds the sites that matter.
        """
        for _ in range(4):
            before = len(self.chat_models)
            self._bind_once(tree)
            if len(self.chat_models) == before:
                return

    def _bind_once(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            # Every segment and bound name of every import. Provider
            # resolution needs the vocabulary of the file, not its dependency
            # graph: `from livingeval import ollama as client` names the
            # provider in the segment, not the package.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.imports.update(alias.name.split("."))
                    self.imports.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.imports.update(node.module.split("."))
                for alias in node.names:
                    self.imports.add(alias.name)
                    if alias.asname:
                        self.imports.add(alias.asname)
            # x = ChatOpenAI(...)  -- and chains like ChatOpenAI(...).bind_tools(...)
            if isinstance(node, ast.Assign):
                ctor = self._ctor_name(node.value) or self._derived_from(node.value)
                if ctor in CHAT_MODEL_CTORS | RUNNABLE_CTORS or ctor == "_derived":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.chat_models.add(target.id)
                        elif isinstance(target, ast.Attribute):
                            self.chat_models.add(target.attr)
            # Literal["a", "b"] and enum classes are the classification signal.
            if isinstance(node, ast.Subscript) and _dotted(node.value)[-1:] == ("Literal",):
                self.enums.append(ast.unparse(node))
            if isinstance(node, ast.ClassDef):
                bases = [_dotted(b)[-1:] for b in node.bases]
                if ("Enum",) in bases or ("StrEnum",) in bases:
                    self.enums.append(node.name)
                if ("BaseModel",) in bases or ("TypedDict",) in bases:
                    self.schema_shapes[node.name] = self._shape(node)
                    self.schema_fields[node.name] = self._fields(node)

    @staticmethod
    def _fields(node: ast.ClassDef) -> list[dict]:
        """Each annotated field: its name, whether it is required, its labels.

        Required means not `Optional[...]`, not `X | None`, and with no default
        -- which is what a schema means by required and what the metric checks.
        Labels are the members of a `Literal[...]`, which is where enum
        membership comes from.
        """
        out: list[dict] = []
        for item in node.body:
            if not isinstance(item, ast.AnnAssign) or not isinstance(item.target, ast.Name):
                continue
            annotation = item.annotation
            optional = False
            if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
                optional = any(isinstance(side, ast.Constant) and side.value is None
                               for side in (annotation.left, annotation.right))
            if (isinstance(annotation, ast.Subscript)
                    and _dotted(annotation.value)[-1:] == ("Optional",)):
                optional = True
            labels: list[str] = []
            if (isinstance(annotation, ast.Subscript)
                    and _dotted(annotation.value)[-1:] == ("Literal",)):
                members = (annotation.slice.elts
                           if isinstance(annotation.slice, ast.Tuple)
                           else [annotation.slice])
                labels = [m.value for m in members
                          if isinstance(m, ast.Constant) and isinstance(m.value, str)]
            out.append({"name": item.target.id,
                        "required": not optional and item.value is None,
                        "labels": labels})
        return out

    @staticmethod
    def _shape(node: ast.ClassDef) -> tuple[int, bool]:
        """Field count, and whether every field is drawn from a fixed label set."""
        fields = [n for n in node.body if isinstance(n, ast.AnnAssign)]
        if not fields:
            return (0, False)
        labelled = all(
            isinstance(f.annotation, ast.Subscript)
            and _dotted(f.annotation.value)[-1:] == ("Literal",)
            for f in fields
        )
        return (len(fields), labelled)

    def _derived_from(self, node: ast.AST | None) -> str | None:
        """"_derived" if this expression is built out of a chat model already
        tracked, otherwise None.

        Two idioms, both ubiquitous and both invisible to `_ctor_name`, which
        only looks for a constructor:

            chain = prompt | llm | parser        # LCEL, a BinOp
            bound = llm.bind_tools(tools)        # a builder rooted at a name

        Missing them means missing `chain.ainvoke(...)`, which in a LangChain
        codebase is where nearly every model call happens.
        """
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            if (self._derived_from(node.left) or self._derived_from(node.right)):
                return "_derived"
            return None
        if isinstance(node, ast.Call):
            if self._ctor_name(node):
                return "_derived"
            return self._derived_from(node.func)
        if isinstance(node, ast.Attribute):
            return self._derived_from(node.value)
        if isinstance(node, ast.Name):
            return "_derived" if node.id in self.chat_models else None
        return None

    def _ctor_name(self, node: ast.AST | None) -> str | None:
        """The chat-model constructor at the root of an expression, if any.

        Unwraps the builder chains these libraries encourage, so that
        `ChatOpenAI(...).bind_tools(t).with_structured_output(S)` still reports
        `ChatOpenAI` -- the name is what later marks invocations of the result
        as call sites.
        """
        while isinstance(node, ast.Call):
            chain = _dotted(node.func)
            if chain and chain[-1] in CHAT_MODEL_CTORS | RUNNABLE_CTORS:
                return chain[-1]
            node = node.func.value if isinstance(node.func, ast.Attribute) else None
        return None

    def walk_functions(self, tree: ast.AST) -> None:
        seen: set[int] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            visitor = _FunctionVisitor(self.chat_models, self.imports)
            for child in node.body:
                visitor.visit(child)
            if not visitor.calls:
                continue
            visitor.evidence.literal_enums = list(self.enums)
            visitor.evidence.schema_shapes = dict(self.schema_shapes)
            visitor.evidence.schema_fields = {k: list(v)
                                              for k, v in self.schema_fields.items()}
            for line, provider, model in visitor.calls:
                if line in seen:
                    continue
                seen.add(line)
                self.sites.append(CallSite(
                    path=self.path, line=line, function=node.name,
                    provider=provider, model=model, evidence=visitor.evidence))

        # Module-level calls, outside any function.
        top = _FunctionVisitor(self.chat_models, self.imports)
        for child in getattr(tree, "body", []):
            if not isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                top.visit(child)
        top.evidence.literal_enums = list(self.enums)
        top.evidence.schema_shapes = dict(self.schema_shapes)
        top.evidence.schema_fields = {k: list(v)
                                      for k, v in self.schema_fields.items()}
        for line, provider, model in top.calls:
            if line not in seen:
                seen.add(line)
                self.sites.append(CallSite(
                    path=self.path, line=line, function="<module>",
                    provider=provider, model=model, evidence=top.evidence))


def scan_file(path: Path) -> list[CallSite]:
    """Every model call in one Python file. Unparseable files yield nothing."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return []
    visitor = _ModuleVisitor(path)
    visitor.collect_bindings(tree)
    visitor.walk_functions(tree)
    return visitor.sites


SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".tox",
             "build", "dist", ".mypy_cache", ".pytest_cache", "site-packages",
             ".eggs", ".ruff_cache", "evals"}


def scan_tree(root: Path, include_tests: bool = False) -> list[CallSite]:
    """Every model call under `root`.

    Tests are excluded by default. A call site in a test is usually a mock or a
    fixture, and generating an eval suite for the mocks rather than the
    application is a memorable way to waste somebody's afternoon.
    """
    sites: list[CallSite] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not include_tests and (
            "test" in path.parts or path.name.startswith("test_")
            or path.name.endswith("_test.py")
        ):
            continue
        # Relative to the repository root, always. These paths are written
        # into generated Python -- as a docstring, and as the dotted module the
        # harness imports -- and an absolute Windows path is neither importable
        # nor even parseable there: "C:\Users\..." is a truncated \U escape
        # and the whole harness fails to compile.
        sites.extend(
            replace(site, path=_relative(site.path, root))
            for site in scan_file(path))
    return sites


def _relative(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path
