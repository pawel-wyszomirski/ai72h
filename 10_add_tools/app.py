from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.asknews import AskNewsSearch

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
    tavily_tool = TavilySearchResults(max_results=2)
    youtube_tool = YouTubeSearchTool()
    wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    asknews_tool = AskNewsSearch(max_results=2)

    tools = [add, tavily_tool, youtube_tool, wikipedia_tool, asknews_tool]
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

    prompt = "Search for youtube video about current coach of polish men football team"
    for step in agent.stream({"messages": [("human", prompt)]}, stream_mode="updates", config=config):
        print(step)
