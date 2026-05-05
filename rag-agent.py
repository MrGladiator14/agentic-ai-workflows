"""RAG-enhanced agent for document retrieval and task execution."""

import json
import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from vector_db import VectorDB

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

vdb = VectorDB()
search = DuckDuckGoSearchRun()

class State(TypedDict):
    """Agent state with RAG support."""
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int
    use_rag: bool
    rag_ctx: List[str]

def rag_decider(state: State) -> State:
    """Decide whether to use RAG based on the goal and tasks.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with RAG decision.
    """
    sys = """You are a RAG decision agent. Analyze the user's goal and determine if it would benefit from 
    retrieval-augmented generation using a knowledge base containing information about NLP, GANs, and Transformers.
    
    Consider using RAG if:
    - The goal involves technical explanations about NLP, deep learning, or AI concepts
    - The user asks for definitions, explanations, or comparisons
    - The topic is related to machine learning architectures or techniques
    - The query requires factual knowledge about AI/ML concepts
    
    Respond ONLY with a JSON object: {"use_rag": true/false, "reason": "brief explanation"}
    """
    
    msgs = [
        SystemMessage(content=sys),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    
    resp = llm.invoke(msgs).content.strip()
    resp = resp.replace("```json", "").replace("```", "").strip()
    
    try:
        dec = json.loads(resp)
        use_rag = dec.get("use_rag", False)
        reason = dec.get("reason", "")
    except json.JSONDecodeError:
        use_rag = False
        reason = "Failed to parse decision"
    
    print(f"[RAG Decider] Use RAG: {use_rag} | Reason: {reason}")
    
    return {**state, "use_rag": use_rag, "rag_ctx": []}

def rag_retriever(state: State) -> State:
    """Retrieve relevant documents using RAG.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with retrieved RAG context.
    """
    if not state["use_rag"]:
        return state
    
    print("[RAG Retriever] Retrieving relevant documents...")
    
    queries = [state["goal"]]
    
    if state.get("tasks"):
        queries.extend(state["tasks"][:3])
    
    rag_ctx = []
    seen_docs = set()
    
    for q in queries:
        try:
            results = vdb.search(q, k=2)
            for res in results:
                doc_text = res['doc']
                if doc_text not in seen_docs:
                    rag_ctx.append(doc_text)
                    seen_docs.add(doc_text)
        except Exception as e:
            print(f"Error searching for query '{q}': {e}")
            continue
    
    print(f"[RAG Retriever] Retrieved {len(rag_ctx)} relevant documents")
    
    return {**state, "rag_ctx": rag_ctx}

def planner(state: State) -> State:
    """Enhanced planner with RAG context consideration.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with planned tasks.
    """
    sys = """You are a planning agent. Break the user's goal into at most 3 concrete, actionable tasks. 
    If RAG context is provided, consider it when creating tasks to ensure they're well-informed.
    Respond ONLY with a valid JSON array of strings. No preamble, no markdown."""
    
    human_msg = f"Goal: {state['goal']}"
    
    if state.get("rag_ctx"):
        ctx_text = "\n\nRelevant Context:\n" + "\n".join([f"- {ctx[:200]}..." for ctx in state["rag_ctx"][:3]])
        human_msg += ctx_text
    
    msgs = [
        SystemMessage(content=sys),
        HumanMessage(content=human_msg)
    ]
    
    resp = llm.invoke(msgs).content.strip()
    resp = resp.replace("```json", "").replace("```", "").strip()
    
    try:
        clean = resp.replace("```json", "").replace("```", "").strip()
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        tasks = [resp]
    
    print(f"[Planner] Generated {len(tasks)} tasks:")
    for i, t in enumerate(tasks):
        print(f" {i+1}. {t}")
    
    return {**state, "tasks": tasks}

def executor(state: State) -> State:
    """Enhanced executor with RAG integration.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with execution results.
    """
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"\nYour previous attempt was rejected. Critique: {state['critique']}\nImprove your output accordingly"
    
    rag_ctx_str = ""
    if state.get("rag_ctx"):
        rag_ctx_str = "\n\nRelevant Knowledge Base Information:\n" + "\n".join([f"• {ctx}" for ctx in state["rag_ctx"]])
    
    for task in state["tasks"]:
        sys = f"""You are an execution agent. Complete the task below thoroughly. 
        Use web search if you need current information.{critique_ctx}
        {rag_ctx_str if rag_ctx_str else ""}"""
        
        search_ctx = ""
        try:
            search_res = search.run(task[:100])
            search_ctx = f"\nWeb search result for context:\n{search_res[:800]}"
        except:
            pass
        
        human_msg = f"Task: {task}{search_ctx}"
        
        msgs = [
            SystemMessage(content=sys),
            HumanMessage(content=human_msg)
        ]
        
        res = llm.invoke(msgs).content
        results.append(res)
        print(f"[Executor] Task: {task[:60]}...\nResult: {res[:120]}...")
    
    return {**state, "results": results}

def verifier(state: State) -> State:
    """Enhanced verifier with RAG context consideration.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with verification results.
    """
    if state["iterations"] >= 3:
        print("[Verifier] Max iterations reached - force approving.")
        state["approved"] = True
        return state

    combined_res = "\n".join(
        [f"Task {i+1}: {t}\nResult: {r}" for i, (t, r) in enumerate(zip(state["tasks"], state["results"]))]
    )

    sys_prompt = (
        "You are a quality verifier. Evaluate the results against the original goal using this rubric:\n"
        "1. Completeness: Does it fully address the goal? (0-0.4)\n"
        "2. Accuracy: Is the information correct and specific? (0-0.3)\n"
        "3. Clarity: Is it well-structured and clear? (0-0.3)\n"
        "Sum the scores for a total between 0.0 and 1.0.\n"
        "Respond ONLY as JSON: {\"score\": 0.85, \"approved\": true, \"critique\": \"...\"}"
    )

    human_msg = f"Original goal: {state['goal']}\n\nResults:\n{combined_res}"
    
    if state.get("rag_ctx"):
        human_msg += f"\n\nRAG Context Used: {len(state['rag_ctx'])} documents retrieved"

    msgs = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=human_msg)
    ]

    raw = llm.invoke(msgs).content.strip()
    clean = raw.replace("```json", "").replace("```", "").strip()
    
    try:
        verdict = json.loads(clean)
        approved = verdict.get("approved", False)
        critique = verdict.get("critique", "")
        score = verdict.get("score", 0.0)
    except Exception:
        approved, critique, score = False, raw, 0.0

    print(f"[Verifier] Score: {score:.2f} | Approved: {approved}")

    state["approved"] = approved
    state["critique"] = critique
    state["score"] = score
    return state

def should_continue(state: State) -> str:
    """Determine whether to continue or end the workflow.
    
    Args:
        state: Current agent state.
        
    Returns:
        Next node name or END.
    """
    return "planner" if not state["approved"] else END

graph = StateGraph(State)

graph.add_node("rag_decider", rag_decider)
graph.add_node("rag_retriever", rag_retriever)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)

graph.set_entry_point("rag_decider")

graph.add_edge("rag_decider", "rag_retriever")
graph.add_edge("rag_retriever", "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")
graph.add_conditional_edges("verifier", should_continue)

app = graph.compile()

def run_workflow(goal: str):
    """Run the enhanced workflow with RAG support.
    
    Args:
        goal: User goal to process.
        
    Returns:
        Final state after workflow completion.
    """
    print(f"\n{'='*50}")
    print(f"Starting Enhanced Workflow with RAG Support")
    print(f"Goal: {goal}")
    print(f"{'='*50}\n")
    
    init_state: State = {
        "goal": goal,
        "tasks": [],
        "results": [],
        "critique": "",
        "approved": False,
        "iterations": 0,
        "use_rag": False,
        "rag_ctx": []
    }
    
    final_state = app.invoke(init_state)
    
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    print(f"RAG Used: {final_state['use_rag']}")
    if final_state['use_rag']:
        print(f"RAG Documents Retrieved: {len(final_state['rag_ctx'])}")
    print(f"Score: {final_state.get('score', 'N/A')}")
    print(f"Iterations: {final_state['iterations']}")
    print(f"\nTasks and Results:")
    for i, (task, res) in enumerate(zip(final_state["tasks"], final_state["results"])):
        print(f"\nTask {i+1}: {task}")
        print(f"Result {i+1}: {res}")
    
    return final_state

if __name__ == "__main__":
    test_goals = [
        "Find the latest AI research papers",
        "In which week was BERT and GPT architectures introduced?"
    ]
    
    for goal in test_goals:
        res = run_workflow(goal)
        print(f"\n{'#'*80}\n")