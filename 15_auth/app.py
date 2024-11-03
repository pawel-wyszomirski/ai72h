import uuid
import os
import chainlit as cl
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import YouTubeSearchTool, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools.asknews import AskNewsSearch
from langchain_core.messages import AIMessageChunk, ToolMessage


load_dotenv(override=True)


@tool
async def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b


def build_agent():
    system_message = SystemMessage(
        content="You are a helpful assistant. Use all the tools to answer the question"
    )

    llm = ChatOpenAI(model="gpt-4o", streaming=True)
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


@cl.on_chat_start
async def on_chat_start():
    agent = build_agent()
    config = {
        "configurable": {
            "thread_id": uuid.uuid4()
        }
    }
    cl.user_session.set("config", config)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent = cl.user_session.get("agent")
    config = cl.user_session.get("config")

    msg = cl.Message(content="")

    async for chunk, metadata in agent.astream({"messages": [("human", message.content)]}, config=config, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            await msg.stream_token(token=chunk.content)
        elif isinstance(chunk, ToolMessage):
            async with cl.Step(name=chunk.name) as step:
                step.output = chunk.content

    await msg.send()

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    if (username, password) == ("piotr", os.environ["APP_PASSWORD"]):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
