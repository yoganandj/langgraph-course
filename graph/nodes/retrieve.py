from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    print(f"Retrieving documents for question: {state['question']}")
    question = state["question"]
    docs = retriever.invoke(question)
    return {"documents": docs, "question": question}


        