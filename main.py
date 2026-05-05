import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langgraph.graph import StateGraph, END
# from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

# Load environment variables from .env file
load_dotenv()

# Shared LLM Instance 
# Note: Use os.getenv("GROQ_API_KEY") for security instead of hardcoding
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY") 
)

# Shared State Schema
class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int

# Search Tool
search = DuckDuckGoSearchRun()

def planner(state: AgentState) -> AgentState:
    system = """You are a planning agent. Break the user's goal into
at most 5 concrete, actionable tasks. Respond ONLY with a
valid JSON array of strings. No preamble, no markdown."""

    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    response = llm.invoke(messages).content.strip()
    response.replace("```json", "").replace("```", "").strip()
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        tasks = [response]

    print(f"[Planner] Generated {len(tasks)} tasks:")
    for i, t in enumerate(tasks): print(f" {i+1}. {t}")

    return {**state, "tasks": tasks}

def executor(state: AgentState) -> AgentState:
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"\nYour previous attempt was rejected. Critique: {state['critique']}\nImprove your output accordingly"
    
    for task in state["tasks"]:
        system = f"""You are an execution agent. Complete the task below thoroughly. Use web search if you need current information.{critique_ctx}"""
        
        # try web search for research tasks
        search_ctx = ""
        try:
            search_result = search.run(task[:100])
            search_ctx = f"\nWeb search result for context:\n{search_result[:800]}"
        except:
            pass
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=f"Task: {task}{search_ctx}")
        ]
        
        result = llm.invoke(messages).content
        results.append(result)
        print(f"[Executor] Task: {task[:60]}...\nResult: {result[:120]}...")
    
    return {**state, "results": results}

import json
from langchain_core.messages import SystemMessage, HumanMessage

def verifier(state: AgentState) -> AgentState:
    if state["iterations"] >= 3:
        print("[Verifier] Max iterations reached - force approving.")
        state["approved"] = True
        return state

    combined_results = "\n".join(
        [f"Task {i+1}: {t}\nResult: {r}" for i, (t, r) in enumerate(zip(state["tasks"], state["results"]))]
    )

    system_prompt = (
        "You are a quality verifier. Evaluate the results against the original goal using this rubric:\n"
        "1. Completeness: Does it fully address the goal? (0-0.4)\n"
        "2. Accuracy: Is the information correct and specific? (0-0.3)\n"
        "3. Clarity: Is it well-structured and clear? (0-0.3)\n"
        "Sum the scores for a total between 0.0 and 1.0.\n"
        "Respond ONLY as JSON: {\"score\": 0.85, \"approved\": true, \"critique\": \"...\"}"
    )

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Original goal: {state['goal']}\n\nResults:\n{combined_results}")
    ]

    raw = llm.invoke(messages).content.strip()
    
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

def should_continue(state: AgentState) -> str:
    return "planner" if not state["approved"] else END

graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)
graph.set_entry_point("planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")
graph.add_conditional_edges("verifier", should_continue)
app = graph.compile()

initial_state: AgentState = {
    "goal": "Find the latest AI research papers",
    "tasks": [],
    "results": [],
    "critique": "",
    "approved": False,
    "iterations": 0
    }  

final_state = app.invoke(initial_state)
print("\n=== Result ===")
for i, (task,result) in enumerate(zip(final_state["tasks"], final_state["results"])):
    print(f"Task {i+1}: {task}")
    print(f"task executor {i+1}: {result}")
