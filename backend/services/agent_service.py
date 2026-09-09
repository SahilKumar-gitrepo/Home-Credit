"""
Agent Service — LangGraph agentic workflow integration.

Wraps the existing agentic_underwriting.py as a FastAPI service.
Gracefully degrades if Ollama is unavailable.

IMPORTANT:
The LLM is NOT the decision maker.
The deterministic underwriting system is the source of truth.
The agent ONLY interprets and presents established evidence.
"""

import json
import logging
import sys
from typing import Optional

from backend.config import settings
from backend.schemas.underwriting import AgentChatResponse

logger = logging.getLogger(__name__)

# Add src/ to path
_src_path = str(settings.model_path.parent.parent / "src")
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)


class AgentService:
    """Singleton service for LangGraph agent interactions."""

    _instance: Optional["AgentService"] = None

    def __init__(self):
        self._llm = None
        self._llm_available = False
        self._loaded = False

    @classmethod
    def get_instance(cls) -> "AgentService":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        """Try to load LLM. Called lazily or at startup."""
        if self._loaded:
            return

        try:
            if settings.llm_provider == "ollama":
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(
                    model=settings.ollama_model,
                    base_url=settings.ollama_base_url,
                    temperature=0,
                )
                # Test connectivity
                self._llm.invoke("ping")
                self._llm_available = True
                logger.info("Ollama LLM available: %s", settings.ollama_model)
            else:
                logger.warning("Unknown LLM provider: %s", settings.llm_provider)
                self._llm_available = False

        except Exception as e:
            logger.warning(
                "LLM unavailable (%s). Deterministic underwriting still works. "
                "Error: %s",
                settings.llm_provider,
                e,
            )
            self._llm_available = False

        self._loaded = True

    @property
    def llm_available(self) -> bool:
        return self._llm_available

    def chat(
        self,
        applicant_id: int,
        question: str,
        underwriting_context: Optional[dict] = None,
    ) -> AgentChatResponse:
        """
        Answer a question about an applicant using LLM + tools.
        Falls back to deterministic summary if LLM is unavailable.
        """
        if not self._loaded:
            self.load()

        if not self._llm_available or self._llm is None:
            return self._fallback_response(applicant_id, question, underwriting_context)

        try:
            return self._agent_response(applicant_id, question, underwriting_context)
        except Exception as e:
            logger.warning("Agent call failed: %s", e)
            return self._fallback_response(applicant_id, question, underwriting_context)

    def _agent_response(
        self, applicant_id: int, question: str, underwriting_context: Optional[dict]
    ) -> AgentChatResponse:
        """Call the LangGraph agent with tools."""
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_core.tools import tool
        from langgraph.graph import StateGraph, END
        from langgraph.prebuilt import ToolNode
        from langgraph.graph.message import add_messages
        from typing import Annotated, TypedDict

        from backend.services.underwriting_service import (
            run_underwriting, calculate_shap, get_risk_tier, make_decision
        )
        from backend.services.policy_service import policy_service
        from backend.services.model_service import model_service

        # Build tools scoped to this applicant
        @tool
        def get_underwriting_result(applicant_id: int) -> str:
            """
            Get the deterministic underwriting result for an applicant.
            Returns PD, risk tier, decision, and policy reason.
            """
            try:
                result = run_underwriting(applicant_id)
                return json.dumps({
                    "applicant_id": applicant_id,
                    "probability_of_default": result.probability_of_default,
                    "risk_tier": result.risk_tier,
                    "decision": result.decision,
                    "policy_reason": result.policy_reason,
                }, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        @tool
        def get_shap_explanation(applicant_id: int) -> str:
            """
            Get SHAP explanation for an applicant.
            SHAP values describe feature contribution to model output only.
            They do NOT establish causality.
            """
            try:
                applicant = model_service.get_applicant(applicant_id)
                increasing, reducing = calculate_shap(applicant)
                return json.dumps({
                    "risk_increasing_factors": increasing[:5],
                    "risk_reducing_factors": reducing[:5],
                    "note": "SHAP values describe model behavior, not causal effects.",
                }, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        @tool
        def search_rbi_policy(question: str) -> str:
            """
            Search the RBI Guidelines on Digital Lending knowledge base.
            Returns regulatory evidence with source, authority, page, and content.
            """
            try:
                results = policy_service.search(question, k=3)
                evidence = [{
                    "source": r.source,
                    "authority": r.authority,
                    "title": r.title,
                    "page": r.page,
                    "content": r.content[:600],
                } for r in results]
                return json.dumps(evidence, default=str)
            except Exception as e:
                return json.dumps({"error": str(e)})

        tools = [get_underwriting_result, get_shap_explanation, search_rbi_policy]
        llm_with_tools = self._llm.bind_tools(tools)

        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]

        system_prompt = f"""You are an AI credit underwriting analyst assistant.

You are answering questions about applicant {applicant_id}.

EVIDENCE RULES (MANDATORY):
1. PD and decision MUST come from get_underwriting_result tool only.
2. SHAP values MUST come from get_shap_explanation tool only.
3. Regulatory statements MUST come from search_rbi_policy tool only.
4. NEVER invent income, loan amounts, credit history, or personal details.
5. NEVER call SHAP values causal. Say "increased/decreased the model output".
6. NEVER claim RBI compliance or legal approval.
7. DAYS_BIRTH is stored as negative days — do NOT say "negative age".
8. Encoded categorical values (integers) CANNOT be interpreted without mapping.
9. Keep answers factual, evidence-based, and concise.

Question about applicant {applicant_id}: {question}"""

        def agent_node(state: AgentState):
            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = llm_with_tools.invoke(messages)
            return {"messages": [response]}

        def route(state: AgentState):
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return END

        graph = StateGraph(AgentState)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", ToolNode(tools))
        graph.set_entry_point("agent")
        graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
        graph.add_edge("tools", "agent")
        app = graph.compile()

        result = app.invoke({"messages": [HumanMessage(content=question)]})
        final_message = result["messages"][-1]
        answer = final_message.content if hasattr(final_message, "content") else str(final_message)

        return AgentChatResponse(
            applicant_id=applicant_id,
            question=question,
            answer=answer,
            llm_available=True,
        )

    def _fallback_response(
        self, applicant_id: int, question: str, underwriting_context: Optional[dict]
    ) -> AgentChatResponse:
        """Return a deterministic fallback when LLM is unavailable."""
        if underwriting_context:
            ctx = underwriting_context
            answer = (
                f"[LLM Unavailable — Deterministic Summary]\n\n"
                f"Applicant: {applicant_id}\n"
                f"Probability of Default: {ctx.get('probability_of_default', 'N/A'):.2%}\n"
                f"Risk Tier: {ctx.get('risk_tier', 'N/A')}\n"
                f"Decision: {ctx.get('decision', 'N/A')}\n"
                f"Reason: {ctx.get('policy_reason', 'N/A')}\n\n"
                f"Your question: '{question}'\n\n"
                f"The AI assistant requires Ollama to be running locally "
                f"({settings.ollama_base_url}). "
                f"The deterministic underwriting result above is the authoritative answer. "
                f"Start Ollama with: ollama serve"
            )
        else:
            answer = (
                f"[LLM Unavailable]\n\n"
                f"The AI assistant requires Ollama running at {settings.ollama_base_url}.\n"
                f"Start Ollama: ollama serve\n"
                f"Pull model: ollama pull {settings.ollama_model}"
            )

        return AgentChatResponse(
            applicant_id=applicant_id,
            question=question,
            answer=answer,
            llm_available=False,
        )


# Module-level singleton
agent_service = AgentService.get_instance()
