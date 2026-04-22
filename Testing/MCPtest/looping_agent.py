import asyncio
import anthropic
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

async def run_agent():
    client = anthropic.Anthropic()

    # Tell the MCP client how to start your server
    server_params = StdioServerParameters(
        command="python",
        args=["server.py"]
    )

    # Connect to your MCP server
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:

            # Ask the server what tools it has
            await session.initialize()
            tools_response = await session.list_tools()

            print("Tools discovered from MCP server:")
            for tool in tools_response.tools:
                print(f"  - {tool.name}: {tool.description}")

            # Convert MCP tool format to Anthropic API format
            tools = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.inputSchema
                }
                for tool in tools_response.tools
            ]

            # Now run the agent loop - same as before
            messages = [
                {"role": "user", "content": "What is 42 + 58? And what is the weather in Melbourne?"}
            ]

            print("\nRunning agent...\n")

            while True:
                response = client.messages.create(
                    model="claude-opus-4-5",
                    max_tokens=1024,
                    tools=tools,
                    messages=messages
                )

                messages.append({"role": "assistant", "content": response.content})

                if response.stop_reason == "end_turn":
                    print("Final answer:")
                    print(response.content[0].text)
                    break

                elif response.stop_reason == "tool_use":
                    for block in response.content:
                        if block.type == "tool_use":
                            print(f"Agent calling tool: {block.name} with {block.input}")

                            # Execute the tool via MCP server
                            # Note: we're not calling the function directly
                            # we're asking the SERVER to run it
                            result = await session.call_tool(block.name, block.input)
                            result_text = result.content[0].text

                            print(f"Tool returned: {result_text}\n")

                            messages.append({
                                "role": "user",
                                "content": [{
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": result_text
                                }]
                            })

asyncio.run(run_agent())