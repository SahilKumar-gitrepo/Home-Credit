"""
Agentic Credit Underwriting System

Architecture:

User
  |
  v
LangGraph Agent
  |
  +---- analyze_applicant()
  |
  +---- explain_applicant()
  |
  +---- search_rbi_policy()
  |
  v
Deterministic underwriting workflow
  |
  +---- XGBoost PD model
  |
  +---- SHAP explanation
  |
  +---- Decision engine
  |
  v
Final evidence-grounded assessment
"""


import json
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from typing import Annotated, TypedDict

from underwriting_agent import run_underwriting_workflow
from policy_retriever import search_policy
from feature_descriptions import describe_feature


# ============================================================
# PATH SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


# ============================================================
# STATE
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    applicant_id: int


# ============================================================
# TOOL 1 — ANALYZE APPLICANT
# ============================================================

@tool
def analyze_applicant(applicant_id: int) -> str:
    """
    Analyze an applicant using the deterministic
    credit underwriting workflow.

    Returns:
        - probability of default
        - risk tier
        - deterministic decision
        - policy reason
    """

    result = run_underwriting_workflow(applicant_id)

    assessment = result["final_assessment"]

    output = {
        "applicant_id": applicant_id,

        "probability_of_default": assessment[
            "probability_of_default"
        ],

        "risk_tier": assessment[
            "risk_tier"
        ],

        "decision": assessment[
            "decision"
        ],

        "policy_reason": assessment[
            "policy_reason"
        ]
    }

    return json.dumps(
        output,
        default=str,
        indent=2
    )


# ============================================================
# TOOL 2 — EXPLAIN APPLICANT
# ============================================================

@tool
def explain_applicant(applicant_id: int) -> str:
    """
    Explain the applicant's model prediction using SHAP.

    SHAP values indicate contribution to the model output.
    They do NOT establish causal relationships.
    """

    result = run_underwriting_workflow(applicant_id)

    assessment = result["final_assessment"]

    risk_increasing = []

    for factor in assessment.get(
        "risk_increasing_factors",
        []
    ):

        risk_increasing.append(
            describe_feature(
                factor["feature"],
                factor["value"],
                factor["shap"]
            )
        )

    risk_reducing = []

    for factor in assessment.get(
        "risk_reducing_factors",
        []
    ):

        risk_reducing.append(
            describe_feature(
                factor["feature"],
                factor["value"],
                factor["shap"]
            )
        )

    output = {

        "applicant_id":
            applicant_id,

        "probability_of_default":
            assessment[
                "probability_of_default"
            ],

        "risk_tier":
            assessment[
                "risk_tier"
            ],

        "decision":
            assessment[
                "decision"
            ],

        "risk_increasing_factors":
            risk_increasing,

        "risk_reducing_factors":
            risk_reducing,

        "interpretation_note": (
            "SHAP values describe the contribution of "
            "features to the model output. They should "
            "not be interpreted as causal explanations."
        )
    }

    return json.dumps(
        output,
        default=str,
        indent=2
    )


# ============================================================
# TOOL 3 — SEARCH RBI POLICY
# ============================================================

@tool
def search_rbi_policy(question: str) -> str:
    """
    Search the local RBI policy knowledge base.

    Returns policy evidence with:
        - source
        - page
        - authority
        - content
    """

    results = search_policy(
        question,
        k=4
    )

    evidence = []

    for result in results:

        metadata = result.metadata

        evidence.append({

            "source":
                metadata.get(
                    "source_file",
                    "Unknown"
                ),

            "page":
                metadata.get(
                    "page",
                    "Unknown"
                ),

            "authority":
                metadata.get(
                    "authority",
                    "Unknown"
                ),

            "title":
                metadata.get(
                    "title",
                    "Unknown"
                ),

            "content":
                result.page_content
        })

    return json.dumps(
        evidence,
        default=str,
        indent=2
    )


# ============================================================
# TOOLS
# ============================================================

TOOLS = [
    analyze_applicant,
    explain_applicant,
    search_rbi_policy
]


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0
)

llm_with_tools = llm.bind_tools(
    TOOLS
)


# ============================================================
# SYSTEM PROMPT
# ============================================================

def build_system_prompt(applicant_id: int) -> str:

    return f"""
You are an AI credit underwriting analyst.

You are analyzing applicant {applicant_id}.

Your task is to produce an evidence-grounded underwriting
assessment.

You have three tools:

1. analyze_applicant
   - Provides the deterministic ML + policy result.

2. explain_applicant
   - Provides SHAP-based model evidence.

3. search_rbi_policy
   - Retrieves evidence from the local RBI policy knowledge base.

============================================================
IMPORTANT EVIDENCE RULES
============================================================

1. The Probability of Default (PD) MUST come from the
   trained credit-risk model.

2. The final decision MUST come from the deterministic
   underwriting policy.

3. Do NOT change, override, or reinterpret the deterministic
   decision.

4. SHAP values describe how features contributed to the
   model output.

5. SHAP values DO NOT establish causality.

6. Never claim that a feature "caused" default risk.

7. Never invent applicant information.

8. Never invent financial information.

9. Never invent regulatory requirements.

10. Regulatory claims must be supported by retrieved
    policy evidence.

11. Keep the following categories separate:

    MODEL EVIDENCE
    - PD
    - SHAP factors
    - model risk tier

    POLICY EVIDENCE
    - deterministic underwriting rules
    - retrieved RBI evidence

    FINAL DECISION
    - APPROVE
    - REFER
    - DECLINE

12. Encoded categorical variables must NOT be interpreted
    as their semantic category unless an explicit mapping
    is available.

    For example:

    NAME_EDUCATION_TYPE = 0

    must NOT be described as "low education".

13. DAYS_BIRTH is represented as negative days in the
    Home Credit dataset.

    For example:

    DAYS_BIRTH = -14394

    must NOT be described as "negative age".

14. If you do not have enough evidence to interpret a feature,
    say:

    "The feature's encoded value cannot be semantically
     interpreted without its category mapping."

15. Do not recommend loan terms, interest rates, amounts,
    repayment periods, or other conditions unless such
    information is explicitly provided by the tools.

16. Do not claim that approval means the applicant is
    guaranteed to repay.

17. Do not claim that the model is fair, unbiased,
    causal, or legally compliant unless evidence is
    explicitly available.

============================================================
REQUIRED FINAL FORMAT
============================================================

Applicant:
<applicant ID>

Probability of Default:
<PD percentage>

Risk Tier:
<risk tier>

Decision:
<deterministic decision>

Policy Reason:
<deterministic policy reason>

MODEL EVIDENCE

Risk-Increasing Factors:
- feature
- value
- SHAP contribution
- short factual description

Risk-Reducing Factors:
- feature
- value
- SHAP contribution
- short factual description

Important:
Explain SHAP contributions as model behavior,
not causal effects.

REGULATORY EVIDENCE

Only include RBI evidence actually retrieved from the
policy tool.

For every important regulatory statement, provide:

Source:
Page:
Authority:
Evidence:

FINAL ASSESSMENT

Give a short factual summary.

Do not introduce new facts.

============================================================

Applicant ID:
{applicant_id}
"""


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(state: AgentState):

    applicant_id = state["applicant_id"]

    system_message = SystemMessage(
        content=build_system_prompt(
            applicant_id
        )
    )

    messages = [
        system_message
    ] + state["messages"]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [
            response
        ]
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_agent(state: AgentState):

    last_message = state[
        "messages"
    ][-1]

    if hasattr(
        last_message,
        "tool_calls"
    ) and last_message.tool_calls:

        return "tools"

    return END


# ============================================================
# GRAPH
# ============================================================

graph = StateGraph(
    AgentState
)

graph.add_node(
    "agent",
    agent_node
)

graph.add_node(
    "tools",
    ToolNode(TOOLS)
)

graph.set_entry_point(
    "agent"
)

graph.add_conditional_edges(
    "agent",
    route_after_agent,
    {
        "tools": "tools",
        END: END
    }
)

graph.add_edge(
    "tools",
    "agent"
)

app = graph.compile()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    applicant_id = 370920

    print("=" * 70)
    print("AGENTIC CREDIT UNDERWRITING SYSTEM")
    print("=" * 70)

    print()
    print(
        f"Analyzing applicant: {applicant_id}"
    )

    print()
    print("Running LangGraph agent...")
    print()

    initial_state = {

        "messages": [
            HumanMessage(
                content=(
                    f"""
Perform a complete underwriting assessment
for applicant {applicant_id}.

You should:

1. Analyze the applicant.
2. Retrieve SHAP model evidence.
3. Retrieve relevant RBI policy evidence.
4. Produce the final structured assessment.

Do not invent information.
"""
                )
            )
        ],

        "applicant_id":
            applicant_id
    }

    result = app.invoke(
        initial_state
    )

    final_message = result[
        "messages"
    ][-1]

    print("=" * 70)
    print("FINAL AGENT ASSESSMENT")
    print("=" * 70)

    print()
    print(
        final_message.content
    )

    print()
    print("=" * 70)