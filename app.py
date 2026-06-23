from dotenv import load_dotenv
load_dotenv()

from importlib.metadata import version
from langchain_google_genai import ChatGoogleGenerativeAI

print(f"LangChain Core Version: {version('langchain-core')}")
print(f"LangGraph Version: {version('langgraph')}")
print(f"Google GenAI Version: {version('langchain-google-genai')}")

def main():
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    response = llm.invoke("Say 'Setup complete' in one word")
    print(response.content)

if __name__ == "__main__":
    main()