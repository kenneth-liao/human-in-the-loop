from goop.graph import build_graph
from langchain_core.messages import AIMessageChunk, HumanMessage, ToolCallChunk
from typing import AsyncGenerator, Any
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
import json
from langchain_core.runnables.config import RunnableConfig
from colorama import Fore, Back, Style


async def process_tool_call_chunk(chunk: ToolCallChunk):
    """Process a tool call chunk and return a formatted string."""
    tool_call_str = ""

    tool_name = chunk.get("name", "")
    args = chunk.get("args", "")

    if tool_name:
        tool_call_str += f"\n\n< TOOL CALL: {tool_name} >\n\n"
    if args:
        tool_call_str += args

    return tool_call_str


async def stream_graph_responses(
        input: dict[str, Any] | Command,
        graph: CompiledStateGraph,
        **kwargs
        ) -> AsyncGenerator[str, None]:
    """Asynchronously stream the result of the graph run.

    Args:
        input: The input to the graph.
        graph: The compiled graph.
        **kwargs: Additional keyword arguments.

    Returns:
        str: The final LLM or tool call response
    """
    async for message_chunk, metadata in graph.astream(
        input=input,
        stream_mode="messages",
        **kwargs
        ):
        if isinstance(message_chunk, AIMessageChunk):
            if message_chunk.response_metadata:
                finish_reason = message_chunk.response_metadata.get("finish_reason", "")
                if finish_reason == "tool_calls":
                    yield "\n\n"

            if message_chunk.tool_call_chunks:
                tool_chunk = message_chunk.tool_call_chunks[0]
                tool_call_str = await process_tool_call_chunk(tool_chunk)
                yield tool_call_str
                
            else:
                # Ensure content is always a string
                content = message_chunk.content
                if isinstance(content, str):
                    yield content
                elif isinstance(content, list):
                    # Convert list content to string representation
                    yield str(content)
                else:
                    # Fallback for any other type
                    yield str(content)


async def main():
    try:
        graph = await build_graph()

        # Checkpointing and a thread_id are required for human-in-the-loop in Langgraph
        config = RunnableConfig(
            recursion_limit=25,
            configurable = {
                "thread_id": "1"
            }
        )
        
        # YOLO mode will always skip human review for protected tools.
        yolo_mode = False

        # Initial input
        graph_input = {
            "messages": [
                HumanMessage(content="Briefly introduce yourself and offer to help me.")
            ],
            "yolo_mode": yolo_mode
        }

        while True:
            # Run the graph until it interrupts
            print(f" ---- 🤖 Goop ---- \n")
            async for response in stream_graph_responses(graph_input, graph, config=config):
                print(Fore.CYAN + response + Style.RESET_ALL, end="", flush=True)

            # Get the thread state after the run
            thread_state = graph.get_state(config=config)

            # Check if there are any interrupts
            while thread_state.interrupts:
                
                # if interrupt, collect input and handle resume
                for interrupt in thread_state.interrupts:
                    print("\n ----- ✅ / ❌ Human Approval Required ----- \n")
                    interrupt_json_str = json.dumps(interrupt.value, indent=2, ensure_ascii=False, default=str)
                    print(Fore.YELLOW + interrupt_json_str + Style.RESET_ALL, flush=True)
                    print("\n Please specify whether you want to reject, continue, update, or provide feedback.", flush=True)

                    action = ""
                    data = None
                    # Validate the action is one the allowed options
                    while action not in ["reject", "continue", "update", "feedback", "exit"]:
                        print("\nInvalid action. Please try again.\n")
                        action = input("Action (reject, continue, update, feedback): ")
                    
                    if action == "exit":
                            print("\n\nExit command received. Exiting...\n\n")
                            return

                    # If additional data is required, collect it
                    if action in ["update", "feedback"]:
                        data = input("Data: ")

                    # Resume the graph with the human input
                    print(f" ----- 🤖 Assistant ----- \n")
                    async for response in stream_graph_responses(Command(resume={"action": action, "data": data}), graph, config=config):
                        print(Fore.CYAN + response + Style.RESET_ALL, end="", flush=True)

                    thread_state = graph.get_state(config=config)

            user_input = input("\n\nUser: ")
            if user_input.lower() in ["exit", "quit"]:
                print("\n\nExit command received. Exiting...\n\n")
                break
            graph_input = {
                "messages": [
                    HumanMessage(content=user_input)
                ],
                "yolo_mode": yolo_mode
            }

            print(f"\n\n ----- 🥷 Human ----- \n\n{user_input}\n")

    except Exception as e:
        print(f"Error: {type(e).__name__}: {str(e)}")
        raise


if __name__ == "__main__":
    import asyncio
    import nest_asyncio
    nest_asyncio.apply()

    asyncio.run(main())
