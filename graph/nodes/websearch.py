import sys
from pathlib import Path
from typing import Any, Dict

# Add project root to path when running as script
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    print("---RUNNING WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]
    
    tavily_search = web_search_tool.invoke({"query": question})
    print(f"---WEB SEARCH RESULTS: {tavily_search}---")
    joined_tavily_results = "\n\n".join([doc["content"] for doc in tavily_search])
    web_results = Document(page_content=joined_tavily_results)
    if documents is None:
        documents = [web_results]
    else:
        documents.append(web_results)
        
    return {
        "question": question,
        "documents": documents
    }
    

if __name__ == "__main__":
    # Example usage
    state = GraphState({
        "question": "agent memory",
        "documents": None,
    })
    result = web_search(state)
    print(result)