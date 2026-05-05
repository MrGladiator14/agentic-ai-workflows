import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

# LangChain & LangGraph Imports
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_groq import ChatGroq

# Import our vector database
from vector_db import VectorDatabase

# Load environment variables from .env file
load_dotenv()

# Shared LLM Instance 
llm = ChatGroq(
    model="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY") 
)

# Initialize Vector Database
vector_db = VectorDatabase()

# Enhanced State Schema with RAG support
class AgentState(TypedDict):
    goal: str
    tasks: List[str]
    results: List[str]
    critique: str
    approved: bool
    iterations: int
    use_rag: bool
    rag_context: List[str]

# Search Tool
search = DuckDuckGoSearchRun()

def rag_decider(state: AgentState) -> AgentState:
    """Decide whether to use RAG based on the goal and tasks"""
    system = """You are a RAG decision agent. Analyze the user's goal and determine if it would benefit from 
    retrieval-augmented generation using a knowledge base containing information about NLP, GANs, and Transformers.
    
    Consider using RAG if:
    - The goal involves technical explanations about NLP, deep learning, or AI concepts
    - The user asks for definitions, explanations, or comparisons
    - The topic is related to machine learning architectures or techniques
    - The query requires factual knowledge about AI/ML concepts
    
    Respond ONLY with a JSON object: {"use_rag": true/false, "reason": "brief explanation"}
    """
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=f"Goal: {state['goal']}")
    ]
    
    response = llm.invoke(messages).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    
    try:
        decision = json.loads(response)
        use_rag = decision.get("use_rag", False)
        reason = decision.get("reason", "")
    except json.JSONDecodeError:
        use_rag = False
        reason = "Failed to parse decision"
    
    print(f"[RAG Decider] Use RAG: {use_rag} | Reason: {reason}")
    
    return {**state, "use_rag": use_rag, "rag_context": []}

def rag_retriever(state: AgentState) -> AgentState:
    """Retrieve relevant documents using RAG"""
    if not state["use_rag"]:
        return state
    
    print("[RAG Retriever] Retrieving relevant documents...")
    
    # Create search queries based on the goal
    queries = [state["goal"]]
    
    # Add queries from tasks if available
    if state.get("tasks"):
        queries.extend(state["tasks"][:3])  # Limit to first 3 tasks
    
    rag_context = []
    seen_docs = set()
    
    for query in queries:
        try:
            results = vector_db.search(query, k=2)
            for result in results:
                doc_text = result['document']
                # Avoid duplicate documents
                if doc_text not in seen_docs:
                    rag_context.append(doc_text)
                    seen_docs.add(doc_text)
        except Exception as e:
            print(f"Error searching for query '{query}': {e}")
            continue
    
    print(f"[RAG Retriever] Retrieved {len(rag_context)} relevant documents")
    
    return {**state, "rag_context": rag_context}

def planner(state: AgentState) -> AgentState:
    """Enhanced planner with RAG context consideration"""
    system = """You are a planning agent. Break the user's goal into at most 5 concrete, actionable tasks. 
    If RAG context is provided, consider it when creating tasks to ensure they're well-informed.
    Respond ONLY with a valid JSON array of strings. No preamble, no markdown."""
    
    human_message = f"Goal: {state['goal']}"
    
    # Add RAG context if available
    if state.get("rag_context"):
        context_text = "\n\nRelevant Context:\n" + "\n".join([f"- {ctx[:200]}..." for ctx in state["rag_context"][:3]])
        human_message += context_text
    
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=human_message)
    ]
    
    response = llm.invoke(messages).content.strip()
    response = response.replace("```json", "").replace("```", "").strip()
    
    try:
        clean = response.replace("```json", "").replace("```", "").strip()
        tasks = json.loads(clean)
    except json.JSONDecodeError:
        tasks = [response]
    
    print(f"[Planner] Generated {len(tasks)} tasks:")
    for i, t in enumerate(tasks): 
        print(f" {i+1}. {t}")
    
    return {**state, "tasks": tasks}

def executor(state: AgentState) -> AgentState:
    """Enhanced executor with RAG integration"""
    results = []
    critique_ctx = ""
    if state["critique"]:
        critique_ctx = f"\nYour previous attempt was rejected. Critique: {state['critique']}\nImprove your output accordingly"
    
    # Prepare RAG context string
    rag_context_str = ""
    if state.get("rag_context"):
        rag_context_str = "\n\nRelevant Knowledge Base Information:\n" + "\n".join([f"• {ctx}" for ctx in state["rag_context"]])
    
    for task in state["tasks"]:
        system = f"""You are an execution agent. Complete the task below thoroughly. 
        Use web search if you need current information.{critique_ctx}
        {rag_context_str if rag_context_str else ""}"""
        
        # Try web search for research tasks
        search_ctx = ""
        try:
            search_result = search.run(task[:100])
            search_ctx = f"\nWeb search result for context:\n{search_result[:800]}"
        except:
            pass
        
        human_message = f"Task: {task}{search_ctx}"
        
        messages = [
            SystemMessage(content=system),
            HumanMessage(content=human_message)
        ]
        
        result = llm.invoke(messages).content
        results.append(result)
        print(f"[Executor] Task: {task[:60]}...\nResult: {result[:120]}...")
    
    return {**state, "results": results}

def verifier(state: AgentState) -> AgentState:
    """Enhanced verifier with RAG context consideration"""
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

    human_message = f"Original goal: {state['goal']}\n\nResults:\n{combined_results}"
    
    # Add RAG context to verification if available
    if state.get("rag_context"):
        human_message += f"\n\nRAG Context Used: {len(state['rag_context'])} documents retrieved"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_message)
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
    """Determine whether to continue or end the workflow"""
    return "planner" if not state["approved"] else END

# Build the enhanced workflow graph
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("rag_decider", rag_decider)
graph.add_node("rag_retriever", rag_retriever)
graph.add_node("planner", planner)
graph.add_node("executor", executor)
graph.add_node("verifier", verifier)

# Set entry point
graph.set_entry_point("rag_decider")

# Add edges
graph.add_edge("rag_decider", "rag_retriever")
graph.add_edge("rag_retriever", "planner")
graph.add_edge("planner", "executor")
graph.add_edge("executor", "verifier")
graph.add_conditional_edges("verifier", should_continue)

# Compile the graph
app = graph.compile()

def run_workflow(goal: str):
    """Run the enhanced workflow with RAG support"""
    print(f"\n{'='*50}")
    print(f"Starting Enhanced Workflow with RAG Support")
    print(f"Goal: {goal}")
    print(f"{'='*50}\n")
    
    initial_state: AgentState = {
        "goal": goal,
        "tasks": [],
        "results": [],
        "critique": "",
        "approved": False,
        "iterations": 0,
        "use_rag": False,
        "rag_context": []
    }  
    
    final_state = app.invoke(initial_state)
    
    print(f"\n{'='*50}")
    print("Final Results")
    print(f"{'='*50}")
    print(f"RAG Used: {final_state['use_rag']}")
    if final_state['use_rag']:
        print(f"RAG Documents Retrieved: {len(final_state['rag_context'])}")
    print(f"Score: {final_state.get('score', 'N/A')}")
    print(f"Iterations: {final_state['iterations']}")
    print(f"\nTasks and Results:")
    for i, (task, result) in enumerate(zip(final_state["tasks"], final_state["results"])):
        print(f"\nTask {i+1}: {task}")
        print(f"Result {i+1}: {result}")
    
    return final_state

if __name__ == "__main__":
    # Example usage
    test_goals = [
        "Explain how self-attention works in transformers",
        "What are the key differences between GANs and VAEs?",
        "Find the latest AI research papers",  # This should not use RAG
        "Compare BERT and GPT architectures"
    ]
    
    for goal in test_goals:
        result = run_workflow(goal)
        print(f"\n{'#'*80}\n")