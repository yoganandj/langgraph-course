from langchain_core.prompts import ChatPromptTemplate

from pydantic import BaseModel, Field
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0)


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    
    binary_score: str = Field(description="Documents are relevant to the question? (yes/no)")
    
     
structured_llm_grader = llm.with_structured_output(GradeDocuments)

system = """You are a grader assesing the relevance of retrieved documents to a given question. \n
if the documents contains keyword(s) or semantic menaing related to the question, grade it relevant. \n
Give a binary score of 'yes' or 'no' for the relevance check. \n"""

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved Documents: \n\n{documents}\n\n User Question: {question}")
    ]
)

retrieval_grader = grade_prompt | structured_llm_grader
   
    
    