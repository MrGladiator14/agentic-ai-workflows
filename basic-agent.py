"""Basic agent for task planning, execution, and verification."""

import json
import os
from typing import TypedDict, List

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY")
)

search = DuckDuckGoSearchRun()

class State(TypedDict):
    """Agent state for task management."""
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int

def planner(state: State) -> State:
    """Break goal into actionable tasks.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with planned tasks.
    """
    sys = """You are a planning agent. Break the user's goal into
    at most 5 concrete, actionable tasks. Respond ONLY with a
    valid JSON array of strings. No preamble, no markdown."""

    msgs = [
        SystemMessage(content=sys),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    resp = llm.invoke(msgs).content.strip()
    resp.replace("```json", "").replace("```", "").strip()
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
    """Execute tasks with web search support.
    
    Args:
        state: Current agent state.
        
    Returns:
        Updated state with execution results.
    """
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"\nYour previous attempt was rejected. Critique: {state['critique']}\nImprove your output accordingly"
    
    for task in state["tasks"]:
        sys = f"""You are an execution agent. Complete the task below thoroughly. Use web search if you need current information.{critique_ctx}"""
        
        search_ctx = ""
        try:
            search_res = search.run(task[:100])
            search_ctx = f"\nWeb search result for context:\n{search_res[:800]}"
        except:
            pass
        
        msgs = [
            SystemMessage(content=sys),
            HumanMessage(content=f"Task: {task}{search_ctx}")
        ]
        
        res = llm.invoke(msgs).content
        results.append(res)
        print(f"[Executor] Task: {task[:60]}...\nResult: {res[:120]}...")
    
    return {**state, "results": results}

def verifier(state: State) -> State:
    """Verify task completion quality.
    
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

    msgs = [
        SystemMessage(content=sys_prompt),
        HumanMessage(content=f"Original goal: {state['goal']}\n\nResults:\n{combined_res}")
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
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)
graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")
graph.add_conditional_edges("verifier", should_continue)
app = graph.compile()

def main():
    """Run the basic agent workflow."""
    init_state: State = {
        "goal": "Find the latest AI research papers",
        "tasks": [],
        "results": [],
        "critique": "",
        "approved": False,
        "iterations": 0
    }
    
    final_state = app.invoke(init_state)
    print("\n=== Result ===")
    for i, (task, res) in enumerate(zip(final_state["tasks"], final_state["results"])):
        print(f"Task {i+1}: {task}")
        print(f"Task executor {i+1}: {res}")

if __name__ == "__main__":
    main()
