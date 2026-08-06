from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph, END, START
from nodes import run_agent_reasoning, tool_node

load_dotenv()
AGENT_REASON = "agent_reason" # agent_reason node and this is starting node
ACT="act" # tool node
LAST=-1

def should_continue(state: MessagesState) -> str:
    """
    Determines whether the agent should continue reasoning or end the process.

    Args:
        state (MessagesState): The current state of the messages.

    Returns:
        str: The next node to transition to, either ACT or END.
    """
    if not state["messages"][LAST].tool_calls:
        return END
    return ACT

flow = StateGraph(MessagesState)

flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.add_node(ACT, tool_node)

flow.add_edge(START, AGENT_REASON)
flow.add_conditional_edges(AGENT_REASON, should_continue,{
    ACT:ACT,
    END:END,
})

flow.add_edge(ACT, AGENT_REASON)

app = flow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")






if __name__ == "__main__":
    print("Hello ReAct ")
    res = app.invoke({"messages":[HumanMessage(content="What is the temperature in Tokyo? List it and then triple it.")]} )
    print(res["messages"][LAST].content)
