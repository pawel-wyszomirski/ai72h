from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(override=True)


@tool
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b


def build_agent():
    system_message = SystemMessage(
        content="You are a helpful assistant. Use all the tools to answer the question"
    )

    llm = ChatOpenAI(model="gpt-4o")
    tools = [add]
    memory = MemorySaver()

    langgraph_agent_executor = create_react_agent(
        llm,
        tools,
        state_modifier=system_message,
        checkpointer=memory
    )
    return langgraph_agent_executor


if __name__ == "__main__":
    agent = build_agent()
    config = {
        "configurable": {
            "thread_id": "123"
        }
    }
    prompt = "Hi, my name is Peter"
    for step in agent.stream({"messages": [("human", prompt)]}, stream_mode="updates", config=config):
        print(step)

    prompt = "What is my name?"
    for step in agent.stream({"messages": [("human", prompt)]}, stream_mode="updates", config=config):
        print(step)
