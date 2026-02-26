from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_ollama import ChatOllama
class State(TypedDict):
    messages: Annotated[list[str], add_messages]
graph_builder = StateGraph(State)

llm=ChatOllama(
    model="llama3.2:latest",
    temperature=0
    )

def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

graph_builder.add_node("chatbot",chatbot)

graph_builder.set_entry_point("chatbot")
graph_builder.set_finish_point("chatbot")
graph=graph_builder.compile()
# try:
#     png_data = graph.get_graph().draw_mermaid_png(max_retries=5, retry_delay=2.0)
#     with open("graph.png", "wb") as f:
#         f.write(png_data)
#     print("Graph image saved to graph.png")
# except Exception as e:
#     print(f"Could not generate graph image: {e}")
#     # Fallback: save the Mermaid text so you can paste it at https://mermaid.live
#     mermaid_text = graph.get_graph().draw_mermaid()
#     with open("graph.mmd", "w") as f:
#         f.write(mermaid_text)
#     print("Saved Mermaid diagram text to graph.mmd — paste it at https://mermaid.live to view")

while True:
    user_input = input("You: ")
    if user_input.lower() in {"exit", "quit"}:
        break
    for event in graph.stream({"messages": [("user", user_input)]}):
        for value in event.values():
            print("Assistant:",value['messages'][0].content)
