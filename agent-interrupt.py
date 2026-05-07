import json
import operator
from typing import Annotated, List, TypedDict, Literal, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv 

from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

class AgentState(TypedDict):
    task: str
    research_notes: Annotated[List[str], operator.add]
    draft: str
    next_node: str
    retry_count: int
    revision_feedback: str

class Router(BaseModel): 
    """Decide which worker to call next.""" 
    next_worker: Literal["researcher", "writer", "FINISH"] = Field(description="The next node to act") 
    instructions: str = Field(description="Specific instructions for the worker")

llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
structured_llm = llm.with_structured_output(Router)
search_tool = TavilySearchResults(k=2)

def researcher(state: AgentState):
    print("\n[Node: Researcher] Digging for info...")
    query = state['task']
    results = search_tool.invoke(query)
    return {"research_notes": [str(results)], "retry_count": state['retry_count'] + 1}

def writer(state: AgentState):
    print("\n[Node: Writer] Composing report...")
    context = "\n".join(state["research_notes"])
    res = llm.invoke(f"Write a professional report on {state['task']} using these notes: {context}")
    return {"draft": res.content}

def supervisor(state: AgentState):
    print("\n[Node: Supervisor] Reviewing State...")
    
    if state['retry_count'] >= 3:
        if state['research_notes'] and not state['draft']:
            return {"next_node": "writer", "revision_feedback": "Max research reached. Use current notes."}
        return {"next_node": "FINISH", "revision_feedback": "Exceeded max attempts."}
    
    num_notes = len(state['research_notes'])
    draft_status = "empty" if not state['draft'] else "complete"
    
    prompt = f"""You are an orchestrator. 
    TASK: {state['task']}
    RESEARCH ATTEMPTS: {state['retry_count']}
    NOTES COLLECTED: {num_notes}
    DRAFT STATUS: {draft_status}
    
    DECISION RULES (follow strictly):
    - If NOTES COLLECTED is 0: respond with next_worker='researcher'
    - If NOTES COLLECTED > 0 AND draft is empty: respond with next_worker='writer'
    - If draft exists and is non-empty: respond with next_worker='FINISH'
    
    Respond ONLY with valid JSON. Do not ask for more research if notes exist."""
    
    try:
        decision = structured_llm.invoke(prompt)
        next_node = decision.next_worker
        instructions = decision.instructions
    except Exception as e:
        print(f"Error in supervisor decision: {e}")
        if not state['research_notes']:
            next_node = "researcher"
            instructions = "No notes found. Research needed."
        elif not state['draft']:
            next_node = "writer"
            instructions = "Notes available. Time to write."
        else:
            next_node = "FINISH"
            instructions = "Draft complete."
    
    return {
        "next_node": next_node,
        "revision_feedback": instructions
    }

builder = StateGraph(AgentState)
builder.add_node("supervisor", supervisor)
builder.add_node("researcher", researcher)
builder.add_node("writer", writer)

builder.set_entry_point("supervisor")

builder.add_conditional_edges(
    "supervisor",
    lambda x: x['next_node'],
    {
        "researcher": "researcher",
        "writer": "writer",
        "FINISH": END
    },
)

builder.add_edge("researcher", "supervisor")
builder.add_edge("writer", "supervisor")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "v1"}}
initial_input = {
    "task": "Impact of LPU architecture on AI inference speeds", 
    "research_notes": [], 
    "retry_count": 0, 
    "draft": "",
    "next_node": "",
    "revision_feedback": ""
}

print("--- STARTING GRAPH ---")
try:
    for event in graph.stream(initial_input, config, stream_mode="updates"):
        for node, values in event.items():
            print(f"--- Finished Node: {node} ---")
            if 'draft' in values:
                print(f"Draft: {values['draft'][:200]}...")
except Exception as e:
    print(f"Graph execution error: {e}")

print("\n--- GRAPH COMPLETED ---")
final_state = graph.get_state(config)
print(f"Final draft:\n{final_state.values.get('draft', 'No draft generated')}")