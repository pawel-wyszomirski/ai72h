from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from typing import Type, Optional
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)


load_dotenv(override=True)


@tool
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b


def tool_calling():
    llm = ChatOpenAI(model="gpt-4o")
    tools = [add]
    llm_with_tools = llm.bind_tools(tools)
    result = llm_with_tools.invoke("How much is seven plus twenty-two")
    print(result)


class MultiplyInput(BaseModel):
    a: int = Field(description="First integer")
    b: int = Field(description="Second integer")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"
    args_schema: Type[BaseModel] = MultiplyInput

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("The tool does not support async")


def tool_calling2():
    llm = ChatOpenAI(model="gpt-4o")
    multiply_tool = MultiplyTool()
    tools = [multiply_tool]
    llm_with_tools = llm.bind_tools(tools)
    result = llm_with_tools.invoke("How much is seven multiply by twenty-two")
    print(result)


def tool_calling3():
    llm = ChatOpenAI(model="gpt-4o")
    tools = [add, MultiplyTool()]
    llm_with_tools = llm.bind_tools(tools)
    result = llm_with_tools.invoke("How much is seven plus five? And how much is four multiplied by eight?")
    return (result.tool_calls)


def extract_args(functions, name):
    for func in functions:
        if func['name'] == name:
            return func['args']
    return None


def execute_tools():
    result = tool_calling3()
    add_args = extract_args(result, 'add')
    print(add_args)
    print(add.invoke(add_args))
    multiply_args = extract_args(result, 'multiply')
    print(multiply_args)
    print(MultiplyTool().invoke(multiply_args))


if __name__ == "__main__":
    execute_tools()
