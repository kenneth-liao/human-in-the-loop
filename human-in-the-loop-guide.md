# Comprehensive Guide to Human-in-the-Loop with LangGraph

## Overview

Human-in-the-loop (HITL) in LangGraph allows you to pause graph execution at specific points to collect human input, review decisions, or modify the workflow before continuing. This is essential for building reliable, production-ready AI systems that require human oversight.

## Core Concepts

### 1. The `interrupt()` Function

The foundation of HITL in LangGraph is the `interrupt()` function, which pauses execution and waits for human input:

```python
from langgraph.types import interrupt

def human_node(state):
    # Pause execution and wait for human input
    value = interrupt({
        "text_to_revise": state["some_text"],
        "instructions": "Please review and edit if needed"
    })
    return {"some_text": value}
```

As the graph is executed, when it reaches the `interrupt()` call, it will pause execution and it's up to the application to collect human input and resume the graph execution with the provided input.

### 2. Checkpointing Requirement

**Critical**: HITL requires a checkpointer to save state between interruptions:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph

# Checkpointer is REQUIRED for interrupts
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 3. Thread Management

Each conversation/workflow needs a unique `thread_id` for state isolation:

```python
config = {"configurable": {"thread_id": "unique_thread_id"}}
```

## Implementation Patterns

Try these examples yourself! I've tested all of these implementations so you can simply copy and paste them into a Python file and run them, just make sure you've installed the correct dependencies.

### Pattern 1: Basic Human Review and Edit

```python
from typing import TypedDict
import uuid
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig


class State(TypedDict):
    summary: str

def generate_summary(state: State) -> State:
    return {"summary": "The cat sat on the mat and looked at the stars."}

def human_review_edit(state: State) -> State:
    result = interrupt({
        "task": "Please review and edit the generated summary if necessary.",
        "generated_summary": state["summary"]
    })
    return {"summary": result["edited_summary"]}

def downstream_use(state: State) -> State:
    print(f"✅ Using edited summary: {state['summary']}")
    return state

# Build the graph
builder = StateGraph(State)
builder.add_node("generate_summary", generate_summary)
builder.add_node("human_review_edit", human_review_edit)
builder.add_node("downstream_use", downstream_use)

builder.set_entry_point("generate_summary")
builder.add_edge("generate_summary", "human_review_edit")
builder.add_edge("human_review_edit", "downstream_use")
builder.add_edge("downstream_use", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Execute until interrupt
config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
result = graph.invoke({}, config=config)

# Resume with human input
edited_summary = "The cat lay on the rug, gazing peacefully at the night sky."
final_result = graph.invoke(
    Command(resume={"edited_summary": edited_summary}),
    config=config
)

print(final_result)
```

### Pattern 2: Human Approval/Rejection with Routing

```python
from typing import TypedDict, Literal
import uuid
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig


class State(TypedDict):
    llm_output: str
    decision: str
    final_result: str

def generate_output(state: State) -> State:
    return {"llm_output": "This is a potentially sensitive AI-generated response about financial advice."}

def human_approval(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
    decision = interrupt({
        "question": "Do you approve the following output?",
        "llm_output": state["llm_output"]
    })

    if decision == "approve":
        return Command(goto="approved_path", update={"decision": "approved"})
    else:
        return Command(goto="rejected_path", update={"decision": "rejected"})

def approved_path(state: State) -> State:
    print("✅ Output approved! Proceeding with original response.")
    return {"final_result": f"APPROVED: {state['llm_output']}"}

def rejected_path(state: State) -> State:
    print("❌ Output rejected! Using safe fallback response.")
    return {"final_result": "I cannot provide that type of advice. Please consult a professional."}

# Build the graph
builder = StateGraph(State)
builder.add_node("generate_output", generate_output)
builder.add_node("human_approval", human_approval)
builder.add_node("approved_path", approved_path)
builder.add_node("rejected_path", rejected_path)

builder.set_entry_point("generate_output")
builder.add_edge("generate_output", "human_approval")
builder.add_edge("approved_path", END)
builder.add_edge("rejected_path", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Execute until interrupt
config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
result = graph.invoke({}, config=config)

# Resume with human decision
human_decision = "reject"  # Change to "approve" to test the other path
final_result = graph.invoke(
    Command(resume=human_decision),
    config=config
)

print(final_result)
```

### Pattern 3: Tool Call Review and Modification

```python
from typing import TypedDict, Literal, Dict, Any
import uuid
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig


class State(TypedDict):
    pending_tool_call: Dict[str, Any]
    tool_result: str
    messages: list

def generate_tool_call(state: State) -> State:
    # Simulate an LLM generating a tool call
    tool_call = {
        "name": "send_email",
        "args": {
            "to": "boss@company.com",
            "subject": "Urgent: System Down",
            "body": "The entire system is down and we need immediate action!"
        }
    }
    return {"pending_tool_call": tool_call}

def human_review_node(state: State) -> Command[Literal["call_llm", "run_tool"]]:
    human_review = interrupt({
        "question": "Is this tool call correct?",
        "tool_call": state["pending_tool_call"]
    })

    review_action = human_review.get("action", "continue")
    review_data = human_review.get("data")

    if review_action == "continue":
        return Command(goto="run_tool")
    elif review_action == "update":
        # Update the tool call with human modifications
        updated_tool_call = {
            "name": state["pending_tool_call"]["name"],
            "args": review_data
        }
        return Command(goto="run_tool", update={"pending_tool_call": updated_tool_call})
    elif review_action == "feedback":
        # Send feedback back to LLM for regeneration
        return Command(goto="call_llm", update={"messages": [f"Human feedback: {review_data}"]})
    else:
        return Command(goto="run_tool")

def run_tool(state: State) -> State:
    tool_call = state["pending_tool_call"]
    print(f"🔧 Executing tool: {tool_call['name']}")
    print(f"   Args: {tool_call['args']}")
    return {"tool_result": f"Tool {tool_call['name']} executed successfully"}

def call_llm(state: State) -> State:
    print("🤖 LLM processing feedback and regenerating...")
    # Simulate LLM regenerating based on feedback
    new_tool_call = {
        "name": "send_email",
        "args": {
            "to": "team@company.com",
            "subject": "System Status Update",
            "body": "We're experiencing some technical difficulties and are working on a solution."
        }
    }
    return {"pending_tool_call": new_tool_call}

# Build the graph
builder = StateGraph(State)
builder.add_node("generate_tool_call", generate_tool_call)
builder.add_node("human_review_node", human_review_node)
builder.add_node("run_tool", run_tool)
builder.add_node("call_llm", call_llm)

builder.set_entry_point("generate_tool_call")
builder.add_edge("generate_tool_call", "human_review_node")
builder.add_edge("run_tool", END)
builder.add_edge("call_llm", "human_review_node")

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Execute until interrupt
config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
result = graph.invoke({}, config=config)

# Resume with human decision
human_decision = {
    "action": "update",
    "data": {
        "to": "team@company.com",
        "subject": "System Maintenance",
        "body": "We are performing scheduled maintenance. Service will resume shortly."
    }
}
final_result = graph.invoke(
    Command(resume=human_decision),
    config=config
)

print(final_result)
```

### Pattern 4: Input Validation with Retry Loop

```python
from typing import TypedDict
import uuid
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig


class State(TypedDict):
    age: int
    prompt: str
    attempts: int

def get_valid_age(state: State) -> State:
    prompt = state.get("prompt", "Please enter your age (must be a non-negative integer).")
    attempts = state.get("attempts", 0)

    user_input = interrupt({
        "prompt": prompt,
        "attempts": attempts
    })

    try:
        age = int(user_input)
        if age < 0:
            raise ValueError("Age must be non-negative.")

        print(f"✅ Valid age received: {age}")
        return {"age": age, "attempts": attempts + 1}

    except (ValueError, TypeError):
        new_prompt = f"'{user_input}' is not valid. Please enter a non-negative integer for age."
        print(f"❌ Invalid input: {user_input}")

        # Continue the loop by updating state and going back to the same node
        return {
            "prompt": new_prompt,
            "attempts": attempts + 1,
            "age": -1  # Invalid marker
        }

def check_age_validity(state: State) -> str:
    if state.get("age", -1) >= 0:
        return "process_age"
    else:
        return "get_valid_age"

def process_age(state: State) -> State:
    age = state["age"]
    if age < 18:
        category = "minor"
    elif age < 65:
        category = "adult"
    else:
        category = "senior"

    print(f"🎯 Processing age {age} - Category: {category}")
    return {"age_category": category}

# Build the graph
builder = StateGraph(State)
builder.add_node("get_valid_age", get_valid_age)
builder.add_node("process_age", process_age)

builder.set_entry_point("get_valid_age")
builder.add_conditional_edges("get_valid_age", check_age_validity, ["get_valid_age", "process_age"])
builder.add_edge("process_age", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Execute with validation loop
config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})

# First attempt with invalid input
result = graph.invoke({}, config=config)
result = graph.invoke(Command(resume="not a number"), config=config)

# Second attempt with negative number
result = graph.invoke(Command(resume="-5"), config=config)

# Third attempt with valid input
final_result = graph.invoke(Command(resume="25"), config=config)

print(final_result)
```

### Pattern 5: Tool Wrapper for Automatic HITL

```python
from typing import TypedDict, Callable, Any
import uuid
from langchain_core.tools import BaseTool, tool as create_tool
from langgraph.types import interrupt, Command
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.config import RunnableConfig


class State(TypedDict):
    messages: list
    tool_result: str

# Define a sample tool to wrap
@create_tool
def book_hotel(location: str, checkin: str, nights: int) -> str:
    """Book a hotel reservation."""
    return f"Hotel booked in {location} for {nights} nights starting {checkin}"

@create_tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to} with subject '{subject}'"

def add_human_in_the_loop(tool: Callable | BaseTool) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review."""
    if not isinstance(tool, BaseTool):
        tool = create_tool(tool)

    @create_tool(tool.name, description=tool.description, args_schema=tool.args_schema)
    def call_tool_with_interrupt(**tool_input):
        request = {
            "action_request": {
                "action": tool.name,
                "args": tool_input
            },
            "description": "Please review the tool call"
        }

        response = interrupt(request)

        if response["type"] == "accept":
            return tool.invoke(tool_input)
        elif response["type"] == "edit":
            tool_input = response["args"]
            return tool.invoke(tool_input)
        elif response["type"] == "response":
            return response["args"]
        else:
            raise ValueError(f"Unsupported response type: {response['type']}")

    return call_tool_with_interrupt

def call_tool_node(state: State) -> State:
    # Simulate calling a wrapped tool
    wrapped_tool = add_human_in_the_loop(book_hotel)

    # This will trigger the interrupt for human review
    result = wrapped_tool.invoke({
        "location": "Paris",
        "checkin": "2024-07-01",
        "nights": 3
    })

    return {"tool_result": result}

# Build the graph
builder = StateGraph(State)
builder.add_node("call_tool_node", call_tool_node)

builder.set_entry_point("call_tool_node")
builder.add_edge("call_tool_node", END)

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Execute until interrupt
config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})
result = graph.invoke({}, config=config)

# Resume with human decision
human_decision = {
    "type": "edit",
    "args": {
        "location": "London",  # Human changed the location
        "checkin": "2024-07-15",  # Human changed the date
        "nights": 2  # Human changed the duration
    }
}

final_result = graph.invoke(
    Command(resume=human_decision),
    config=config
)

print(final_result)
```

## Checkpointer Options

### 1. In-Memory (Development/Testing)

```python
from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

### 2. SQLite (Local Persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Synchronous
with SqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

# Asynchronous
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

### 3. PostgreSQL (Production)

```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/db"

# Synchronous
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()  # Run once to create tables
    graph = builder.compile(checkpointer=checkpointer)

# Asynchronous
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

async with AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # await checkpointer.setup()  # Run once to create tables
    graph = builder.compile(checkpointer=checkpointer)
```

### 4. Redis (Distributed/Cloud)

```python
from langgraph.checkpoint.redis import RedisSaver

DB_URI = "redis://localhost:6379"

# Synchronous
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)

# Asynchronous
from langgraph.checkpoint.redis.aio import AsyncRedisSaver

async with AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer:
    graph = builder.compile(checkpointer=checkpointer)
```

## Execution and Resumption

### Basic Execution Pattern

```python
# Initial execution (runs until interrupt)
config = {"configurable": {"thread_id": "unique_id"}}
result = graph.invoke(initial_input, config=config)

# Check if interrupted
if "__interrupt__" in result:
    print("Graph paused for human input:")
    print(result["__interrupt__"])
    
    # Resume with human input
    final_result = graph.invoke(
        Command(resume=human_input_value),
        config=config
    )
```

### Streaming Execution

```python
# Stream until interrupt
for event in graph.stream(initial_input, config, stream_mode="values"):
    print(event)
    if "__interrupt__" in event:
        break

# Resume streaming
for event in graph.stream(Command(resume=human_input), config, stream_mode="values"):
    print(event)
```

### Multiple Interrupts

```python
# Handle multiple interrupts in a single resume
state = graph.get_state(config)
resume_map = {
    interrupt.interrupt_id: f"response_for_{interrupt.value}"
    for interrupt in state.interrupts
}

graph.invoke(Command(resume=resume_map), config=config)
```

## Best Practices

### 1. Side Effects Management

**Important**: Place side effects AFTER interrupts to avoid re-execution:

```python
# ❌ BAD: Side effect before interrupt (will re-execute)
def bad_node(state):
    api_call()  # This will run again on resume
    answer = interrupt("Question?")
    return {"answer": answer}

# ✅ GOOD: Side effect after interrupt
def good_node(state):
    answer = interrupt("Question?")
    api_call(answer)  # Only runs after resume
    return {"answer": answer}

# ✅ GOOD: Side effect in separate node
def interrupt_node(state):
    answer = interrupt("Question?")
    return {"answer": answer}

def side_effect_node(state):
    api_call(state["answer"])
    return state
```

### 2. Error Handling

```python
def robust_human_node(state):
    try:
        result = interrupt({
            "question": "Please provide input",
            "context": state.get("context", {})
        })

        # Validate human input
        if not result or not isinstance(result, dict):
            raise ValueError("Invalid input format")

        return {"processed_input": result}

    except Exception as e:
        # Log error and provide fallback
        print(f"Error in human node: {e}")
        return {"error": str(e), "processed_input": None}
```

### 3. State Management

```python
# Check current state
state_snapshot = graph.get_state(config)
print(f"Current values: {state_snapshot.values}")
print(f"Next nodes: {state_snapshot.next}")
print(f"Pending interrupts: {state_snapshot.interrupts}")

# Resume from specific checkpoint
config_with_checkpoint = {
    "configurable": {
        "thread_id": "thread_1",
        "checkpoint_id": "specific_checkpoint_id"
    }
}
graph.invoke(None, config=config_with_checkpoint)
```

## Functional API Pattern

```python
from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

@task
def step_1(input_query):
    return f"{input_query} processed"

@task
def human_feedback(input_query):
    feedback = interrupt(f"Please provide feedback: {input_query}")
    return f"{input_query} {feedback}"

@task
def step_3(input_query):
    return f"{input_query} finalized"

@entrypoint(checkpointer=MemorySaver())
def workflow(input_query):
    result_1 = step_1(input_query).result()
    result_2 = human_feedback(result_1).result()
    result_3 = step_3(result_2).result()
    return result_3

# Usage
config = {"configurable": {"thread_id": "1"}}
for event in workflow.stream("initial input", config):
    print(event)
```

## Cloud/Production Deployment

### Using LangGraph SDK

```python
# Python SDK
from langgraph_sdk import get_client
from langgraph_sdk.schema import Command

client = get_client(url=DEPLOYMENT_URL)
thread = await client.threads.create()

# Run until interrupt
result = await client.runs.wait(
    thread["thread_id"],
    assistant_id,
    input={"some_text": "original text"}
)

# Resume with human input
final_result = await client.runs.wait(
    thread["thread_id"],
    assistant_id,
    command=Command(resume="edited text")
)
```

```javascript
// JavaScript SDK
import { Client } from "@langchain/langgraph-sdk";
const client = new Client({ apiUrl: DEPLOYMENT_URL });

const thread = await client.threads.create();

// Run until interrupt
const result = await client.runs.wait(
    thread["thread_id"],
    assistantID,
    { input: { "some_text": "original text" } }
);

// Resume with human input
const finalResult = await client.runs.wait(
    thread["thread_id"],
    assistantID,
    { command: { resume: "edited text" } }
);
```

## Common Use Cases Summary

1. **Content Review**: LLM generates content → Human reviews/edits → Continue processing
2. **Tool Approval**: Agent wants to call tool → Human approves/modifies → Execute tool
3. **Decision Points**: Multiple paths available → Human chooses direction → Continue on chosen path
4. **Input Validation**: Collect user input → Validate → Retry if invalid → Continue when valid
5. **Quality Control**: Process data → Human quality check → Approve or request changes
6. **Sensitive Operations**: Before executing sensitive actions → Human confirmation → Proceed or abort

## Key Takeaways

- **Always use a checkpointer** - Required for any interrupt functionality
- **Manage thread IDs carefully** - Each conversation needs a unique thread
- **Place side effects after interrupts** - Avoid re-execution issues
- **Handle errors gracefully** - Validate human input and provide fallbacks
- **Choose appropriate checkpointer** - Memory for dev, SQLite/Postgres/Redis for production
- **Use streaming for better UX** - Shows progress and handles interrupts smoothly

This comprehensive guide covers all the major patterns and best practices for implementing human-in-the-loop workflows with LangGraph. The key is always ensuring you have proper checkpointing, thread management, and careful handling of side effects around interrupt points.
