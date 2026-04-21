from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, your agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."

# class secret_calculation_tool(BaseTool):
#     name: str = "secret_calculation_tool"
#     description: str = (
#         "A tool to calculate a secret number based on two input integers. "
#         "The tool takes two whitespace-separated integers as input, performs a secret calculation, and returns the result as a string."
#     )
#     args_schema: Type[BaseModel] = MyCustomToolInput

#     def _run(self, argument: str) -> str:
#         # Parse the input argument to extract the two integers
#         try:
#             num1_str, num2_str = argument.split()
#             num1 = int(num1_str)
#             num2 = int(num2_str)
#         except ValueError:
#             return "Invalid input format. Please provide two whitespace-separated integers."

#         # Perform a calculation
#         result = (num1 - 1)  * (num2 + 1)

#         # Return the result as a string
#         return str(result)