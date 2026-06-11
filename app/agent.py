"""Tool-calling agent with per-session conversation memory."""
import logging
from typing import Dict, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.rag_chain import get_chat_llm
from app.tools import ALL_TOOLS

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are the Smart Business Assistant for a company. You can:\n"
    "1. Answer questions about company policies, products and FAQs by searching "
    "the internal documents (always use the search_company_docs tool for these - "
    "never answer from your own knowledge).\n"
    "2. Create contacts in the HubSpot CRM when the user asks to add/register someone.\n"
    "3. Send emails on the user's behalf.\n\n"
    "Rules:\n"
    "- Before creating a contact, make sure you have at least an email address; "
    "ask for it if missing.\n"
    "- When an email should contain company information (policies, hours, pricing, "
    "etc.), FIRST retrieve it with search_company_docs and use only the retrieved "
    "facts in the email body - never invent company details.\n"
    "- Confirm what you did after using a tool, including any IDs returned.\n"
    "- If a tool reports it is not configured or simulated, tell the user honestly.\n"
    "- Be concise, friendly and professional."
)

# Per-session chat history (in-memory; swap for Redis/DynamoDB in production)
_histories: Dict[str, List[BaseMessage]] = {}
MAX_HISTORY_MESSAGES = 20

_executor = None


def _get_executor() -> AgentExecutor:
    global _executor
    if _executor is None:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder("chat_history", optional=True),
                ("human", "{input}"),
                MessagesPlaceholder("agent_scratchpad"),
            ]
        )
        llm = get_chat_llm()
        agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
        _executor = AgentExecutor(
            agent=agent,
            tools=ALL_TOOLS,
            verbose=True,
            max_iterations=6,
            handle_parsing_errors=True,
        )
    return _executor


def run_agent(message: str, session_id: str = "default") -> str:
    """Run one agent turn, maintaining conversation history per session."""
    history = _histories.setdefault(session_id, [])
    result = _get_executor().invoke({"input": message, "chat_history": history})
    answer = result["output"]

    history.append(HumanMessage(content=message))
    history.append(AIMessage(content=answer))
    if len(history) > MAX_HISTORY_MESSAGES:
        del history[: len(history) - MAX_HISTORY_MESSAGES]
    return answer


def reset_session(session_id: str) -> None:
    _histories.pop(session_id, None)
