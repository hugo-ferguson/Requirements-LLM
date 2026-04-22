from dotenv import load_dotenv
import anthropic

load_dotenv()

client = anthropic.Anthropic()

tools = [
    {
        "name": "get_weather",
        "description": "Gets the current weather for a given city",
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to get weather for"
                }
            },
            "required": ["city"]
        }
    }
]

# --- Step 1: First API call ---
messages = [
    {"role": "user", "content": "What is the weather like in Melbourne?"}
]

response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

print("STEP 1 - Model wants to call:", response.content[0].name)
print("With inputs:", response.content[0].input)

# --- Step 2: We "execute" the tool (fake result for now) ---
tool_use_block = response.content[0]

def get_weather(city):
    # Fake implementation - in reality you'd call a weather API here
    return f"It is 22 degrees and sunny in {city}"

tool_result = get_weather(tool_use_block.input["city"])
print("\nSTEP 2 - We executed the tool, result:", tool_result)

# --- Step 3: Feed the result back into the conversation ---
messages = [
    {"role": "user", "content": "What is the weather like in Melbourne?"},
    {"role": "assistant", "content": response.content},  # model's tool call request
    {"role": "user", "content": [          # our tool result
        {
            "type": "tool_result",
            "tool_use_id": tool_use_block.id,
            "content": tool_result
        }
    ]}
]

# --- Step 4: Second API call - model now has the result and can answer ---
final_response = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=1024,
    tools=tools,
    messages=messages
)

print("\nSTEP 3 - Final answer from model:")
print(final_response.content[0].text)