import os
from langchain_core.messages import BaseMessage
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv

load_dotenv()


@tool
def calculator(expression: str) -> str:
	"""
	A simple calculator that evaluates basic math expressions.
	"""
	try:
		return str(eval(expression))
	except Exception as e:
		return f"Error: {e}"


@tool
def lookup_fact(topic: str) -> str:
	"""
	Looks up a fact from the internal knowledge base. Always use this tool
	before answering questions about: python, langchain, or monash.
	"""
	facts: dict[str, str] = {
		"python": "Python was created by Guido van Rossum and released in 1991.",
		"langchain": "LangChain is a framework for building LLM-powered applications.",
		"monash": "Monash University is based in Melbourne, founded in 1958.",
	}

	query = topic.lower().strip()

	for key, value in facts.items():
		if key in query:
			return value
		
	return f"No facts found for '{topic}'. Try asking about python, langchain, or monash."


def print_response(messages: list[BaseMessage]) -> None:
	"""
	Prints the response messages, including any tool calls and their results.
	"""
	for msg in messages:
		tool_calls = getattr(msg, "tool_calls", [])
		if msg.type == "ai" and tool_calls:
			for tool_call in tool_calls:
				print(f"  [Using Tool] {tool_call['name']}({tool_call['args']})")
		elif msg.type == "tool":
			print(f"  [Result] {msg.name} returned: {msg.content}")
		elif msg.type == "ai" and msg.content:
			print(f"\nA: {msg.content}")


def main() -> None:
	queries = [
		"What is LangChain?",
		"What's 1024 * 768?",
		"Tell me about Monash and calculate how old it is in 2026.",
	]

	llm = init_chat_model(
		os.environ["MODEL"],
		base_url=os.getenv("OLLAMA_URL"),
		temperature=0,
	)

	agent = create_agent(
		model=llm,
		tools=[calculator, lookup_fact],
		system_prompt="You are a helpful assistant. Use tools when needed.",
	)

	for query in queries:
		print(f"Q: {query}")

		result = agent.invoke({"messages": [{"role": "user", "content": query}]})

		print_response(result["messages"])


if __name__ == "__main__":
	main()
