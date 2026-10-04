import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

load_dotenv()
print("GROQ KEY FOUND:", bool(os.getenv("GROQ_API_KEY")))
print("GOOGLE KEY FOUND:", bool(os.getenv("GOOGLE_API_KEY")))

# --------------------------------------------------
# 1. Groq - Primary LLM
# --------------------------------------------------

groq_llm = None

if os.getenv("GROQ_API_KEY"):
    groq_llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0
    )


# --------------------------------------------------
# 2. Gemini - Secondary LLM
# --------------------------------------------------

gemini_llm = None

if os.getenv("GOOGLE_API_KEY"):
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.8-flash",
        temperature=0
    )


# --------------------------------------------------
# 3. Ollama - Local Backup
# --------------------------------------------------

ollama_llm = ChatOllama(
    model="llama3.2",
    temperature=0
)


# --------------------------------------------------
# Generate Answer
# --------------------------------------------------
def generate_answer(question, context=""):

    print("DEBUG - Groq available:", groq_llm is not None)
    print("DEBUG - Gemini available:", gemini_llm is not None)

    prompt = f"""
You are an AI Project Intelligence and Risk Advisor.
Use the project context below to answer the question clearly.

Project Context:
{context}

Question:
{question}

Give a useful and clear answer.
"""

    # ----------------------------------------------
    # Try Groq
    # ----------------------------------------------

    if groq_llm:

        try:
            response = groq_llm.invoke(prompt)

            print("LLM Provider: Groq")

            return response.content

        except Exception as e:

            print("Groq failed:", e)
            print("Trying Gemini...")


    # ----------------------------------------------
    # Try Gemini
    # ----------------------------------------------

    if gemini_llm:

        try:
            response = gemini_llm.invoke(prompt)

            print("LLM Provider: Gemini")

            return response.content

        except Exception as e:

            print("Gemini failed:", e)
            print("Trying Ollama...")


    # ----------------------------------------------
    # Try Ollama
    # ----------------------------------------------

    try:

        response = ollama_llm.invoke(prompt)

        print("LLM Provider: Ollama")

        return response.content

    except Exception as e:

        return f"""
All configured LLM providers failed.

Last error:
{e}
"""