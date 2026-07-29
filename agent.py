"""
LangChain Agent – MedAssist AI
──────────────────────────────
ReAct agent with structured tools for:
  • Stock checking
  • Alternative suggestions
  • General FAQ / RAG answers

Uses the existing hybrid retrieval pipeline under the hood.
Supports streaming for real-time token delivery to the UI.
"""

import sys, pathlib
from queue import Queue
from threading import Thread

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.callbacks import BaseCallbackHandler

from config import (
    GROQ_API_KEYS, GROQ_MODEL,
    LLM_TEMPERATURE, LLM_MAX_TOKENS, STREAMING_ENABLED,
)
from rag_engine import retrieve, _StreamingCallbackHandler

# ═══════════════════════════════════════════════════════════
# Agent Tools
# ═══════════════════════════════════════════════════════════

_key_index = 0


@tool
def check_stock(medicine_name: str) -> str:
    """Check the stock availability of a specific medicine.
    Use this when the user asks if a medicine is available or in stock.
    Returns the medicine name, strength, stock status, and dosage info."""
    docs = retrieve(medicine_name)
    if not docs:
        return f"I do not have information on '{medicine_name}' in our inventory."

    results = []
    for d in docs:
        meta = d["metadata"]
        stock_status = "✅ In Stock" if meta.get("stock", "").lower() == "yes" else "❌ Out of Stock"
        results.append(
            f"**{meta['medicine_name']} {meta['strength']}**\n"
            f"  • Status: {stock_status}\n"
            f"  • Use: {meta['use_case']}\n"
            f"  • Dosage: {meta['dosage']}"
        )
    return "\n\n".join(results)


@tool
def suggest_alternatives(medicine_name: str) -> str:
    """Suggest alternative medicines for a given medicine.
    Use this when the user asks for alternatives, substitutes, or replacements.
    Returns the medicine and its listed alternatives from the database."""
    docs = retrieve(medicine_name)
    if not docs:
        return f"I do not have information on '{medicine_name}' in our inventory."

    results = []
    for d in docs:
        meta = d["metadata"]
        stock_status = "in stock" if meta.get("stock", "").lower() == "yes" else "out of stock"
        alt = meta.get("alternative", "None listed")
        results.append(
            f"**{meta['medicine_name']} {meta['strength']}** ({stock_status})\n"
            f"  • Alternative(s): {alt}\n"
            f"  • Original use: {meta['use_case']}"
        )
    return "\n\n".join(results)


@tool
def answer_faq(question: str) -> str:
    """Answer general medical inventory questions using RAG.
    Use this for any general question about medicines, conditions, dosage,
    or inventory that isn't specifically about stock or alternatives.
    Retrieves relevant medicine data and generates a grounded answer."""
    from rag_engine import generate_answer
    answer, docs = generate_answer(question)
    sources = [d["metadata"]["medicine_name"] for d in docs]
    source_text = ", ".join(sources) if sources else "No specific sources"
    return f"{answer}\n\n[Sources: {source_text}]"


# ═══════════════════════════════════════════════════════════
# Agent System Prompt
# ═══════════════════════════════════════════════════════════

AGENT_SYSTEM_PROMPT = """\
You are **MedAssist AI**, a helpful hospital pharmacy assistant.

You have access to three tools to help users:
1. **check_stock** – Use when the user asks about medicine availability or stock status
2. **suggest_alternatives** – Use when the user asks for alternatives or substitutes
3. **answer_faq** – Use for general questions about medicines, conditions, dosage, etc.

RULES:
- Always use the appropriate tool to get information before answering.
- NEVER fabricate medicine information. Only report what the tools return.
- If a medicine is out of stock, proactively mention alternatives.
- End every response that includes dosage information with: "Please consult a doctor before use."
- Be concise, helpful, and professional.
"""


# ═══════════════════════════════════════════════════════════
# Agent Factory
# ═══════════════════════════════════════════════════════════

TOOLS = [check_stock, suggest_alternatives, answer_faq]


def _get_agent_llm(streaming: bool = False, callbacks=None) -> ChatGroq:
    """Create a ChatGroq instance for the agent."""
    global _key_index
    key = GROQ_API_KEYS[_key_index % len(GROQ_API_KEYS)]
    return ChatGroq(
        api_key=key,
        model=GROQ_MODEL,
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        streaming=streaming,
        callbacks=callbacks or [],
    )


def run_agent(query: str) -> tuple[str, list[dict]]:
    """
    Run the agent on a query (non-streaming).
    Returns (final_answer, retrieved_docs).
    """
    global _key_index
    # First retrieve docs for source display
    docs = retrieve(query)

    # Build the agent with tool-calling
    llm = _get_agent_llm(streaming=False)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    # Agent loop: LLM decides tool calls, we execute them
    for _ in range(5):  # max iterations to prevent infinite loops
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check for tool calls
        if response.tool_calls:
            from langchain_core.messages import ToolMessage
            for tc in response.tool_calls:
                tool_map = {t.name: t for t in TOOLS}
                if tc["name"] in tool_map:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            # No more tool calls → final answer
            return response.content, docs

    return response.content, docs


def run_agent_stream(query: str):
    """
    Run the agent with streaming on the final response.
    Yields:
      - First yield: list[dict] of retrieved docs
      - Subsequent yields: str tokens
    """
    global _key_index
    docs = retrieve(query)
    yield docs

    # Run agent non-streaming for tool-calling phase
    llm = _get_agent_llm(streaming=False)
    llm_with_tools = llm.bind_tools(TOOLS)

    messages = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    # Tool-calling loop (non-streaming)
    tool_phase_response = None
    for _ in range(5):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if response.tool_calls:
            from langchain_core.messages import ToolMessage
            for tc in response.tool_calls:
                tool_map = {t.name: t for t in TOOLS}
                if tc["name"] in tool_map:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        else:
            tool_phase_response = response
            break

    # If tool phase already produced a final answer, stream it token by token
    if tool_phase_response and not STREAMING_ENABLED:
        yield tool_phase_response.content
        return

    if tool_phase_response:
        # Re-generate the final answer with streaming enabled
        # Use the tool results context gathered during agent loop
        token_queue: Queue = Queue()
        handler = _StreamingCallbackHandler(token_queue)

        # Build a fresh prompt with tool context baked in
        # Extract tool results from message history
        context_parts = []
        for msg in messages:
            if hasattr(msg, 'content') and isinstance(msg, ToolMessage if 'ToolMessage' in dir() else type(None)):
                context_parts.append(msg.content)

        # Stream the final answer
        def _invoke():
            try:
                llm_stream = _get_agent_llm(streaming=True, callbacks=[handler])
                llm_stream.invoke(messages)
            except Exception:
                token_queue.put(None)

        thread = Thread(target=_invoke, daemon=True)
        thread.start()

        while True:
            token = token_queue.get()
            if token is None:
                break
            yield token

        thread.join(timeout=5)
    else:
        yield "I wasn't able to process that query. Please try rephrasing."


# ═══════════════════════════════════════════════════════════
# CLI Smoke Test
# ═══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("  MedAssist AI – LangChain Agent Smoke Test")
    print("=" * 60)

    test_cases = [
        ("Stock Check", "Is Amoxicillin in stock?"),
        ("Alternatives", "What are alternatives for Ibuprofen?"),
        ("FAQ / RAG", "What medicine should I take for fever?"),
        ("Unknown", "Do you have Aspirin?"),
    ]

    for label, query in test_cases:
        print(f"\n🔍 [{label}] {query}")
        try:
            answer, docs = run_agent(query)
            print(f"   📋 Answer: {answer[:200]}...")
            print(f"   📦 Sources: {[d['metadata']['medicine_name'] for d in docs]}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
