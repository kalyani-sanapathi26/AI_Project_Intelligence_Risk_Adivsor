from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


def generate_answer(question, context=""):
    prompt = f"""
You are an AI Project Intelligence and Risk Advisor.

Use the project context below to answer the question clearly.

Project Context:
{context}

Question:
{question}

Give a useful and clear answer.
"""

    response = llm.invoke(prompt)
    return response.content