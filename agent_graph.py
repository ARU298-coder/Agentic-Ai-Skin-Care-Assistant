from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from brain_of_the_doctor_groq import brain_of_the_doctor

import os
import json

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# -----------------------------
# 1. SHARED AGENT STATE
# -----------------------------

class AgentState(TypedDict, total=False):
    user_text: str
    image_path: str
    has_image: bool

    image_analysis: str
    risk_level: str
    route: str
    final_response: str

    trace: list[str]


# -----------------------------
# 2. ROUTER NODE
# -----------------------------

def router_node(state: AgentState):

    trace = state.get("trace", [])

    if state.get("has_image"):
        route = "image_analysis"
        trace.append("Router -> Image analysis selected")
    else:
        route = "text_analysis"
        trace.append("Router -> Text analysis selected")

    return {
        "route": route,
        "trace": trace
    }


# -----------------------------
# 3. IMAGE TOOL NODE
# -----------------------------
def image_analysis_node(state: AgentState):
    trace = state.get("trace", [])

    patient_text = state.get("user_text", "")
    image_path = state.get("image_path")

    trace.append("Vision Tool -> Analyzing uploaded skin image")

    analysis = brain_of_the_doctor(
        patient_text=patient_text,
        image_filepath=image_path
    )

    trace.append("Vision Tool -> Analysis completed")

    return {
        "image_analysis": analysis,
        "trace": trace
    }

   
# -----------------------------
# 4. TEXT TOOL NODE
# -----------------------------

def text_analysis_node(state: AgentState):

    trace = state.get("trace", [])
    trace.append("Tool -> Text analysis executed")

    return {
        "trace": trace
    }


# -----------------------------
# 5. TRIAGE NODE
# -----------------------------
def triage_node(state: AgentState):
    trace = state.get("trace", [])

    user_text = state.get("user_text", "")
    image_analysis = state.get("image_analysis", "")

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    prompt = f"""
You are a safety triage component for a skin-care AI assistant.

User description:
{user_text}

Vision tool result:
{image_analysis}

Classify the situation into exactly ONE category:

LOW
UNCERTAIN
RED_FLAG

RED_FLAG means the information suggests potentially urgent symptoms
such as difficulty breathing, facial swelling, severe pain, high fever,
rapid spreading, severe blistering, or another potentially serious situation.

UNCERTAIN means there is not enough reliable information or the image
is irrelevant, unclear, or insufficient.

LOW means no obvious urgent warning signs are present.

Return ONLY valid JSON:

{{
  "risk_level": "LOW or UNCERTAIN or RED_FLAG",
  "reason": "short explanation"
}}
"""

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type": "json_object"}
    )

    result = json.loads(
        response.choices[0].message.content
    )

    risk = result["risk_level"]
    reason = result["reason"]

    trace.append(
        f"AI Triage -> {risk}: {reason}"
    )

    return {
        "risk_level": risk,
        "trace": trace
    }

    text = state.get("user_text", "").lower()
    trace = state.get("trace", [])

    red_flags = [
        "rapidly spreading",
        "severe pain",
        "difficulty breathing",
        "swelling of face",
        "high fever"
    ]

    risk = "LOW"

    for flag in red_flags:
        if flag in text:
            risk = "RED_FLAG"
            break

    trace.append(f"Triage -> Risk level: {risk}")

    return {
        "risk_level": risk,
        "trace": trace
    }


# -----------------------------
# 6. CONDITIONAL DECISION
# -----------------------------

def risk_router(state: AgentState):

    risk = state.get("risk_level")

    if risk == "RED_FLAG":
        return "escalate"

    if risk == "UNCERTAIN":
        return "followup"

    return "safe"


# -----------------------------
# 7. SAFE RESPONSE
# -----------------------------
def safe_response_node(state: AgentState):
    trace = state.get("trace", [])

    image_analysis = state.get(
        "image_analysis",
        "No visual analysis was available."
    )

    trace.append("Safety -> Standard guidance path")

    final_response = (
        f"{image_analysis} "
        "No obvious urgent warning signs were identified from the provided information. "
        "If the symptoms worsen, spread, become painful, or you develop other concerning symptoms, "
        "please seek evaluation from a qualified healthcare professional."
    )

    return {
        "final_response": final_response,
        "trace": trace
    }


def followup_node(state: AgentState):

    trace = state.get("trace", [])
    trace.append("Agent -> Additional information requested")

    return {
        "final_response":
            "I cannot reliably assess the concern from the provided image. "
            "Please upload a clear image of the affected skin area and provide "
            "details such as when it started and whether it is spreading, painful, or itchy.",
        "trace": trace
    }

# -----------------------------
# 8. ESCALATION RESPONSE
# -----------------------------

def escalation_node(state: AgentState):

    trace = state.get("trace", [])
    trace.append("Safety -> Escalation path selected")

    return {
        "final_response":
            "A potential red flag was detected. "
            "Please seek evaluation from a qualified healthcare professional.",
        "trace": trace
    }


# -----------------------------
# 9. BUILD LANGGRAPH
# -----------------------------

builder = StateGraph(AgentState)

builder.add_node("router", router_node)
builder.add_node("image_analysis", image_analysis_node)
builder.add_node("text_analysis", text_analysis_node)
builder.add_node("triage", triage_node)
builder.add_node("safe_response", safe_response_node)
builder.add_node("followup", followup_node)
builder.add_node("escalation", escalation_node)


builder.add_edge(START, "router")


# Router chooses image or text path
builder.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "image_analysis": "image_analysis",
        "text_analysis": "text_analysis"
    }
)


builder.add_edge("image_analysis", "triage")
builder.add_edge("text_analysis", "triage")


# Triage chooses safe or escalation path
builder.add_conditional_edges(
    "triage",
    risk_router,
    {
        "safe": "safe_response",
        "followup": "followup",
        "escalate": "escalation"
    }
)


builder.add_edge("safe_response", END)
builder.add_edge("followup", END)
builder.add_edge("escalation", END)


agent = builder.compile()


# -----------------------------
# 10. TEST
# -----------------------------

if __name__ == "__main__":

    result = agent.invoke({
        "user_text":
            "I have a dry irritated patch after using a new face wash.",
        "image_path": "sample-image.png",
        "has_image": True,
        "trace": []
    })

    print("\nIMAGE ANALYSIS:")
    print(result.get("image_analysis"))

    print("\nFINAL RESPONSE:")
    print(result["final_response"])

    print("\nAGENT TRACE:")

    for step in result["trace"]:
        print("✓", step) 