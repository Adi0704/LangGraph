"""
Multi-Node LangGraph Demo
==========================
This demo shows how to build a graph with:
  - Multiple nodes (classifier, joke_teller, fact_provider, advisor, fallback)
  - Conditional edges (routing based on intent)
  - A shared state that flows through every node

Flow:
  User Input → classifier → (joke | fact | advice | fallback) → END
"""

from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama


# ── State ────────────────────────────────────────────────────────────────
class State(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str  # stores the classified intent


# ── LLM ──────────────────────────────────────────────────────────────────
llm = ChatOllama(model="llama3.2:latest", temperature=0)
creative_llm = ChatOllama(model="llama3.2:latest", temperature=0.9)


# ── Node 1: Classifier ──────────────────────────────────────────────────
def classifier(state: State) -> dict:
    """Classify user intent into: joke, fact, advice, or general."""
    last_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1][1]

    classification_prompt = [
        ("system",
         "You are an intent classifier. Classify the user's message into EXACTLY one of these categories: "
         "joke, fact, advice, general. "
         "Reply with ONLY the single category word, nothing else."),
        ("user", last_msg),
    ]
    response = llm.invoke(classification_prompt)
    intent = response.content.strip().lower()

    # Normalize to valid intents
    if intent not in {"joke", "fact", "advice"}:
        intent = "general"

    print(f"  [Classifier] Detected intent: {intent}")
    return {"intent": intent}


# ── Node 2: Joke Teller ─────────────────────────────────────────────────
def joke_teller(state: State) -> dict:
    """Tell a joke related to the user's message."""
    last_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1][1]

    prompt = [
        ("system", "You are a hilarious comedian. Tell a short, funny joke related to what the user said. Keep it clean and witty."),
        ("user", last_msg),
    ]
    response = creative_llm.invoke(prompt)
    print("  [Joke Teller] Generating joke...")
    return {"messages": [response]}


# ── Node 3: Fact Provider ───────────────────────────────────────────────
def fact_provider(state: State) -> dict:
    """Provide an interesting fact related to the user's message."""
    last_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1][1]

    prompt = [
        ("system", "You are a knowledgeable encyclopedia. Provide a concise, fascinating fact related to the user's topic. Include a 'Did you know?' opener."),
        ("user", last_msg),
    ]
    response = llm.invoke(prompt)
    print("  [Fact Provider] Looking up facts...")
    return {"messages": [response]}


# ── Node 4: Advisor ─────────────────────────────────────────────────────
def advisor(state: State) -> dict:
    """Give helpful advice related to the user's message."""
    last_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1][1]

    prompt = [
        ("system", "You are a wise and empathetic advisor. Give brief, actionable advice on the user's topic. Be supportive and practical."),
        ("user", last_msg),
    ]
    response = llm.invoke(prompt)
    print("  [Advisor] Preparing advice...")
    return {"messages": [response]}


# ── Node 5: Fallback / General Chat ─────────────────────────────────────
def fallback(state: State) -> dict:
    """Handle general conversation."""
    last_msg = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else state["messages"][-1][1]

    prompt = [
        ("system", "You are a friendly, helpful assistant. Respond conversationally to the user."),
        ("user", last_msg),
    ]
    response = llm.invoke(prompt)
    print("  [General Chat] Responding...")
    return {"messages": [response]}


# ── Router function (decides which node runs next) ──────────────────────
def route_by_intent(state: State) -> Literal["joke_teller", "fact_provider", "advisor", "fallback"]:
    """Route to the appropriate node based on classified intent."""
    intent = state.get("intent", "general")
    route_map = {
        "joke": "joke_teller",
        "fact": "fact_provider",
        "advice": "advisor",
    }
    return route_map.get(intent, "fallback")


# ── Build the graph ─────────────────────────────────────────────────────
graph_builder = StateGraph(State)

# Add all nodes
graph_builder.add_node("classifier", classifier)
graph_builder.add_node("joke_teller", joke_teller)
graph_builder.add_node("fact_provider", fact_provider)
graph_builder.add_node("advisor", advisor)
graph_builder.add_node("fallback", fallback)

# Set entry point
graph_builder.set_entry_point("classifier")

# Add conditional edges from classifier → appropriate handler
graph_builder.add_conditional_edges(
    "classifier",
    route_by_intent,
    {
        "joke_teller": "joke_teller",
        "fact_provider": "fact_provider",
        "advisor": "advisor",
        "fallback": "fallback",
    },
)

# All handler nodes go to END
graph_builder.add_edge("joke_teller", END)
graph_builder.add_edge("fact_provider", END)
graph_builder.add_edge("advisor", END)
graph_builder.add_edge("fallback", END)

# Compile
graph = graph_builder.compile()

# Save the graph diagram
mermaid_text = graph.get_graph().draw_mermaid()
with open("multi_node_graph.mmd", "w") as f:
    f.write(mermaid_text)
print("Graph diagram saved to multi_node_graph.mmd (paste at https://mermaid.live to view)\n")

# ── Chat loop ────────────────────────────────────────────────────────────
print("=" * 50)
print("  Multi-Node LangGraph Chatbot")
print("  Try: 'tell me a joke', 'give me a fact about space', 'I need advice on studying'")
print("  Type 'exit' to quit")
print("=" * 50)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in {"exit", "quit"}:
        print("Goodbye!")
        break

    for event in graph.stream({"messages": [("user", user_input)], "intent": ""}):
        for node_name, value in event.items():
            if "messages" in value:
                print(f"\nAssistant: {value['messages'][-1].content}")
