from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage


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

    langgraph_agent_executor = create_react_agent(llm, tools, state_modifier=system_message)
    return langgraph_agent_executor


if __name__ == "__main__":
    agent = build_agent()
    prompt = "Return the height of Empire State building in meters as an integer. Then add 100 to the result. Use a tool to add two numbers"
    for step in agent.stream({"messages": [("human", prompt)]}, stream_mode="updates"):
        print(step)
