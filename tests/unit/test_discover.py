"""The scanner, against code shaped like the code it will meet.

Fixtures are written the way people actually write these applications rather
than reduced to the minimum that trips each rule, because the failure mode that
matters is a scanner that passes its own tests and finds nothing in a real
repository.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from livingeval.discover import scan_file
from livingeval.discover.archetype import classify
from livingeval.plan.taxonomy import Archetype


def write(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    return path


def only(tmp_path: Path, name: str, source: str):
    sites = [classify(s) for s in scan_file(write(tmp_path, name, source))]
    assert len(sites) == 1, f"expected one call site, got {[s.ident for s in sites]}"
    return sites[0]


# ---------------------------------------------------------------------------
# provider detection
# ---------------------------------------------------------------------------


def test_openai_client_call(tmp_path):
    site = only(tmp_path, "app.py", '''
        from openai import OpenAI

        def answer(question: str) -> str:
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": question}],
            )
            return resp.choices[0].message.content
    ''')
    assert site.provider == "openai"
    assert site.model == "gpt-4o-mini"
    assert site.function == "answer"


def test_anthropic_client_call(tmp_path):
    site = only(tmp_path, "app.py", '''
        import anthropic

        def answer(question):
            client = anthropic.Anthropic()
            return client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=512,
                messages=[{"role": "user", "content": question}],
            )
    ''')
    assert site.provider == "anthropic"
    assert site.model == "claude-sonnet-4-5"


def test_langchain_runnable_is_a_call_site(tmp_path):
    """The provider is behind a runnable, so `.invoke` on it is the call."""
    site = only(tmp_path, "chain.py", '''
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini")

        def answer(question):
            return llm.invoke(question)
    ''')
    assert site.provider == "langchain"


def test_ambiguous_chat_resolves_from_imports(tmp_path):
    """`client.chat(...)` names no provider by itself."""
    site = only(tmp_path, "oll.py", '''
        import ollama

        def answer(question):
            return ollama.chat(model="llama3", messages=[{"role": "user", "content": question}])
    ''')
    assert site.provider == "ollama"


def test_ambiguous_chat_without_a_known_import_is_not_a_call_site(tmp_path):
    """Better to miss than to invent. `.chat()` is a common method name."""
    sites = scan_file(write(tmp_path, "chatroom.py", '''
        from myapp.rooms import ChatRoom

        def send(room, text):
            return room.chat(text)
    '''))
    assert sites == []


def test_syntax_errors_are_skipped_not_raised(tmp_path):
    assert scan_file(write(tmp_path, "broken.py", "def f(:\n  pass")) == []


# ---------------------------------------------------------------------------
# archetype classification
# ---------------------------------------------------------------------------


def test_rag_from_retrieval_before_generation(tmp_path):
    site = only(tmp_path, "rag.py", '''
        from openai import OpenAI

        def answer(question, store):
            docs = store.similarity_search(question, k=5)
            context = "\\n".join(d.page_content for d in docs)
            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Answer only from the context provided below."},
                    {"role": "user", "content": f"{context}\\n\\n{question}"},
                ],
            )
            return resp.choices[0].message.content
    ''')
    assert site.archetype == Archetype.RAG.value
    assert site.evidence.retrieval


def test_extraction_from_a_declared_schema(tmp_path):
    """A schema beats a retriever: it yields assertions needing no human input."""
    site = only(tmp_path, "extract.py", '''
        from openai import OpenAI
        from pydantic import BaseModel

        class Invoice(BaseModel):
            total: float
            currency: str

        def extract(text):
            client = OpenAI()
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": text}],
                response_format=Invoice,
            )
    ''')
    assert site.archetype == Archetype.EXTRACTION.value
    assert "Invoice" in site.evidence.schemas


def test_classification_needs_a_schema_and_a_label_set(tmp_path):
    site = only(tmp_path, "route.py", '''
        from typing import Literal
        from openai import OpenAI
        from pydantic import BaseModel

        class Route(BaseModel):
            team: Literal["billing", "technical", "general"]

        def route(email):
            client = OpenAI()
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": email}],
                response_format=Route,
            )
    ''')
    assert site.archetype == Archetype.CLASSIFICATION.value


def test_agent_needs_tools_and_a_loop(tmp_path):
    site = only(tmp_path, "agent.py", '''
        from openai import OpenAI

        def run(task, tools):
            client = OpenAI()
            messages = [{"role": "user", "content": task}]
            for _ in range(10):
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=tools,
                )
                if not resp.choices[0].message.tool_calls:
                    return resp
            return resp
    ''')
    assert site.archetype == Archetype.AGENT.value


def test_tools_without_a_loop_is_not_an_agent(tmp_path):
    """One tool call and no loop is structured output wearing a hat."""
    site = only(tmp_path, "once.py", '''
        from openai import OpenAI

        def lookup(question, tools):
            client = OpenAI()
            return client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": question}],
                tools=tools,
            )
    ''')
    assert site.archetype != Archetype.AGENT.value


def test_unrecognised_site_is_generic_not_dropped(tmp_path):
    """An unevaluated model call is exactly what the tool exists to surface."""
    site = only(tmp_path, "plain.py", '''
        from openai import OpenAI

        def rewrite(text):
            client = OpenAI()
            return client.chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": text}],
            )
    ''')
    assert site.archetype == Archetype.GENERIC.value
    assert site.confidence == 0.0
    assert site.rationale


def test_ambiguity_lowers_confidence_even_when_signals_are_strong(tmp_path):
    """Two archetypes tied is a site for a human to look at, not a coin toss."""
    site = only(tmp_path, "both.py", '''
        from openai import OpenAI
        from pydantic import BaseModel

        class Answer(BaseModel):
            text: str

        def answer(question, store):
            docs = store.similarity_search(question, k=5)
            client = OpenAI()
            return client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": str(docs) + question}],
                response_format=Answer,
            )
    ''')
    assert site.confidence < 0.3
    assert any("also looks like" in r for r in site.rationale)


def test_rationale_is_always_populated(tmp_path):
    """Every decision has to be explainable; that is the whole contract."""
    site = only(tmp_path, "rag2.py", '''
        from openai import OpenAI

        def answer(q, retriever):
            docs = retriever.get_relevant_documents(q)
            return OpenAI().chat.completions.create(
                model="gpt-4o-mini", messages=[{"role": "user", "content": str(docs)}],
            )
    ''')
    assert site.rationale and all(isinstance(r, str) and r for r in site.rationale)


def test_a_pure_label_schema_is_confidently_classification(tmp_path):
    """Extraction and classification are readings of the same evidence, not
    rival hypotheses. Letting both score would flag a clear-cut case as
    ambiguous and ask the user a question with an obvious answer."""
    site = only(tmp_path, "triage.py", '''
        from typing import Literal
        from openai import OpenAI
        from pydantic import BaseModel

        class Route(BaseModel):
            team: Literal["billing", "technical", "general"]

        def route(email):
            return OpenAI().chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": email}],
                response_format=Route,
            )
    ''')
    assert site.archetype == Archetype.CLASSIFICATION.value
    assert site.confidence >= 0.5


# ---------------------------------------------------------------------------
# how these objects are actually held
# ---------------------------------------------------------------------------
#
# Every test below started as a call site the scanner missed in a real
# repository -- gpt-researcher, ~5k Python lines of LangChain. Before them it
# found two sites in the whole codebase and none of the ones that matter.


def test_a_model_held_on_self_is_found(tmp_path):
    """`self.llm.ainvoke(...)`. Matching only the root of the dotted chain
    finds `llm.invoke()` in a script and misses every call in a class, which
    is where production code keeps these."""
    site = only(tmp_path, "service.py", '''
        from langchain_openai import ChatOpenAI

        class Researcher:
            def __init__(self):
                self.llm = ChatOpenAI(model="gpt-4o-mini")

            async def summarise(self, messages):
                return await self.llm.ainvoke(messages)
    ''')
    assert site.provider == "langchain"
    assert site.function == "summarise"


def test_an_lcel_pipeline_is_a_call_site(tmp_path):
    """`chain = prompt | llm | parser` is the LangChain idiom, and
    `chain.ainvoke(...)` is where nearly every model call in such a codebase
    happens. A scanner that only recognises constructors sees none of them."""
    site = only(tmp_path, "chain.py", '''
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import StrOutputParser

        async def plan(task):
            prompt = ChatPromptTemplate.from_template("{task}")
            model = ChatOpenAI(model="gpt-4o")
            chain = prompt | model | StrOutputParser()
            return await chain.ainvoke({"task": task})
    ''')
    assert site.provider == "langchain"


def test_a_bound_model_is_still_the_model(tmp_path):
    site = only(tmp_path, "tools.py", '''
        from langchain_openai import ChatOpenAI

        async def act(messages, tools):
            llm = ChatOpenAI(model="gpt-4o")
            llm_with_tools = llm.bind_tools(tools)
            return await llm_with_tools.ainvoke(messages)
    ''')
    assert site.provider == "langchain"


def test_a_compiled_graph_is_a_call_site(tmp_path):
    """LangGraph. The thing invoked is not a chat model at all, but the call
    site is `graph.ainvoke(...)` exactly as if it were."""
    site = only(tmp_path, "graph.py", '''
        from langgraph.graph import StateGraph

        async def run(state):
            workflow = StateGraph(dict)
            graph = workflow.compile()
            return await graph.ainvoke(state)
    ''')
    assert site.provider == "langchain"


def test_a_binding_defined_below_its_use_is_still_found(tmp_path):
    """Source order is not definition order. A helper that builds the graph is
    conventionally written below the method that calls it, so a single pass
    reaches `workflow.compile()` before it knows what `workflow` is."""
    site = only(tmp_path, "late.py", '''
        from langgraph.graph import StateGraph

        class Editor:
            async def run(self, state):
                workflow = self._build()
                chain = workflow.compile()
                return await chain.ainvoke(state)

            def _build(self):
                workflow = StateGraph(dict)
                return workflow
    ''')
    assert site.function == "run"
