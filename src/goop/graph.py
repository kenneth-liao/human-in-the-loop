from pydantic import BaseModel
from typing import Annotated, List, Literal
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    AIMessage,
    ToolMessage
)
from langgraph.types import Command, interrupt
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient
import json
from goop.config import mcp_config


class  AgentState(BaseModel):
    """The state of the agent.
    
    Attributes:
        messages: The list of messages in the conversation.
        protected_tools: The list of tools that require human review.
        yolo_mode: Whether to skip human review for protected tools.
    """
    messages: Annotated[List[BaseMessage], add_messages] = []
    protected_tools: List[str] = [
        # "create_directory",
        # "edit_file",
        # "move_file",
        # "write_file"
    ]
    yolo_mode: bool = False


async def build_graph():
    """
    Build the LangGraph application.
    """
    client = MultiServerMCPClient(connections=mcp_config["mcpServers"])
    tools = await client.get_tools()

    llm = ChatOpenAI(
        model="gpt-4.1-mini-2025-04-14",
        temperature=0.1
    ).bind_tools(tools)

    # llm = ChatOllama(
    #     model="qwen3:4b",
    #     temperature=0.1
    # ).bind_tools(tools)

    def assistant_node(state: AgentState) -> AgentState:
        response = llm.invoke(
            [SystemMessage(content="You are Goop, a helpful assistant. You have access to the local filesystem but only within the /projects/workspace directory. You must use paths relative to /projects/workspace for all tools.")] +
            state.messages
            )
        state.messages = state.messages + [response]
        return state
    
    def human_tool_review_node(state: AgentState) -> Command[Literal["assistant_node", "tools"]]:
        last_message = state.messages[-1]

        # Ensure we have a valid AI message with tool calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            raise ValueError("human_tool_review_node called without valid tool calls")

        tool_call = last_message.tool_calls[-1]

        # Stop graph execution at this node and wait for human input
        # The interrupt value can be of any type
        human_review: dict = interrupt({
            "message": "Your input is required for the following tool:",
            "tool_call": tool_call
        })

        # The type of the interrupt value is defined by the interrupt() call but the value passed back to the graph
        # when resuming must match the structure defined here.
        review_action = human_review.get("action")
        review_data = human_review.get("data")

        # Approve the tool call as-is
        if review_action == "continue":
            return Command(goto="tools")

        # Update the tool call arguments created by our Agent, then proceed to call the tool
        elif review_action == "update":
            if review_data is None:
                raise ValueError("update action requires data")

            updated_message = AIMessage(
                content=last_message.content,
                tool_calls=[{
                    "id": tool_call["id"],
                    "name": tool_call["name"],
                    "args": json.loads(review_data)
                }],
                id=last_message.id
            )

            return Command(goto="tools", update={"messages": [updated_message]})

        # Send feedback to the Agent as a tool message (required after a tool call)
        # The Agent can then decide whether to retry the tool call or not
        elif review_action == "feedback":
            if review_data is None:
                raise ValueError("feedback action requires data")

            tool_message = ToolMessage(
                content=review_data,
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
            return Command(goto="assistant_node", update={"messages": [tool_message]})
        
        # Reject the tool call and send a message to the Agent
        elif review_action == "reject":
            tool_message = ToolMessage(
                content="The tool call was rejected by the user, follow up with the user to understand why and how they would like to proceed.",
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
            return Command(goto="assistant_node", update={"messages": [tool_message]})
        
        else:
            # if nothing is passed, assume continue
            return Command(goto="tools")


    def assistant_router(state: AgentState) -> str:
        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            if not state.yolo_mode:
                if any(tool_call["name"] in state.protected_tools for tool_call in last_message.tool_calls):
                    # how do we handle multiple tool calls?
                    return "human_tool_review_node"
            return "tools"
        else:
            return END

    builder = StateGraph(AgentState)

    builder.add_node(assistant_node)
    builder.add_node(human_tool_review_node)
    builder.add_node("tools", ToolNode(tools))

    builder.add_edge(START, "assistant_node")
    builder.add_conditional_edges("assistant_node", assistant_router, ["tools", "human_tool_review_node", END])
    builder.add_edge("tools", "assistant_node")

    return builder.compile(checkpointer=MemorySaver())


async def inspect_graph(graph):
    """
    Visualize the graph using the mermaid.ink API.
    """
    from IPython.display import display, Image
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))


async def main():
    graph = await build_graph()
    await inspect_graph(graph)


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    asyncio.run(main())
