from dotenv import load_dotenv
load_dotenv()

import logging, os
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_together import ChatTogether
from langgraph.prebuilt import create_react_agent
from langchain.chains import RetrievalQA
from langchain_core.messages.ai import AIMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from rag_store import get_vector_store

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    try:
        if provider == "Groq":
            llm = ChatGroq(model=llm_id)
        elif provider == "OpenAI":
            llm = ChatOpenAI(model=llm_id)
        elif provider == "DeepSeek":
            llm = ChatTogether(model=llm_id, api_key=TOGETHER_API_KEY)
        else:
            return {"error": "Invalid model provider."}

        vector_store = get_vector_store()
        if vector_store:
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())
            result = qa_chain.run(query[-1].content)
            return {"response": result}

        tools = [TavilySearchResults(max_results=2)] if allow_search else []

        agent = create_react_agent(
            model=llm,
            tools=tools,
            state_modifier=system_prompt
        )

        response = agent.invoke({"messages": query})
        ai_messages = [m.content for m in response.get("messages", []) if isinstance(m, AIMessage)]

        return {"response": ai_messages[-1]} if ai_messages else {"response": "Sorry, I couldn't generate a response."}

    except Exception as e:
        logger.error("AI agent error: %s", str(e))
        return {"error": "Internal error occurred in AI agent."}
