from mcp.server.fastmcp import FastMCP

# This creates an MCP server
mcp = FastMCP("My First MCP Server")

# This is all you need to expose a tool via MCP
# The decorator handles all the protocol stuff automatically
@mcp.tool()
def add_numbers(a: int, b: int) -> str:
    """Adds two numbers together and returns the result"""
    return f"The result of {a} + {b} is {a + b}"

@mcp.tool()
def get_weather(city: str) -> str:
    """Gets the current weather for a given city"""
    return f"It is 22 degrees and sunny in {city}"

if __name__ == "__main__":
    mcp.run(transport="stdio")