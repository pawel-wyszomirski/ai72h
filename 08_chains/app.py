from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv(override=True)


def basic_llm_chain():
    print("Welcome to the AI course!")
    llm = ChatOpenAI(model="gpt-3")  # try different models eg. gpt-4o, gpt-3.5-turbo
    print(llm.invoke("Write a short poem about pizza"))


def basic_prompts():
    llm = ChatOpenAI(model="gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an assistant that acts as a {role}. Answer only questions about your role"),
        ("user", "{prompt}")
    ])
    chain = prompt | llm | StrOutputParser()
    print(chain.invoke({"prompt": "How to make pizza?", "role": "cook"}))


def longer_chains():
    llm = ChatOpenAI(model="gpt-4o-mini")
    recipe_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a cook that gives dish recipes"),
        ("user", "Give me a recipe for {dish}")
    ])
    # chain = recipe_prompt | llm | StrOutputParser()

    llm2 = ChatOpenAI(model="gpt-4o")
    ingredients_prompt = ChatPromptTemplate.from_template("Based on the recipe: {recipe} list me all ingredients needed for the recipe. Return just the list and nothing else")

    chain2 = (
        recipe_prompt
        | llm
        | StrOutputParser()
        | (lambda input: {"recipe": input})
        | ingredients_prompt
        | llm2
        | StrOutputParser()
        | (lambda input: f"Here are ingredients for the dish: \n {input}")
    )
    print(chain2.invoke({"dish": "pizza"}))


if __name__ == "__main__":
    longer_chains()
