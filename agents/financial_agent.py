from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.tools import Tool
from langchain import hub

from config.settings import settings
from rag_pipeline.vector_store import VectorStoreManager


class FinancialAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.model_name,
            temperature=settings.temperature,
            openai_api_key=settings.openai_api_key
        )

        self.vector_store = VectorStoreManager()
        self.agent_executor = self._create_agent()

    def _create_rag_tool(self):
        """Create RAG tool for financial knowledge"""

        def search_financial_knowledge(query: str) -> str:
            """Search financial knowledge base for relevant information"""
            docs = self.vector_store.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
            return context

        return Tool(
            name="financial_knowledge_search",
            func=search_financial_knowledge,
            description=(
                "Search for financial information including investments, "
                "tax planning, budgeting, and personal finance concepts."
            ),
        )

    def _create_agent(self):
        """Create the agent with tools"""
        tools = [self._create_rag_tool()]

        prompt = hub.pull("hwchase17/openai-functions-agent")

        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=tools,
            prompt=prompt,
        )

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
        )

    def run(self, query: str) -> str:
        """Run the financial agent"""
        system_message = (
            "You are a financial advisor AI agent. "
            "Use the financial knowledge base to provide accurate and helpful advice. "
            "Always explain your reasoning and cite sources when possible."
        )

        full_query = f"{system_message}\n\nUser Query: {query}"
        result = self.agent_executor.invoke({"input": full_query})

        return result["output"]
