# Agentic AI Skin-Care Assistant

A multimodal Agentic AI application that analyzes a user's skin concern using text or voice input together with an uploaded image.

The system uses **LangGraph** to orchestrate a stateful workflow. Instead of sending every request through one fixed LLM pipeline, the agent analyzes tool outputs and dynamically routes the request through different safety paths.

> This project provides general informational skin-care guidance only. It is not a medical diagnostic system and does not replace professional medical care.

---

## Agentic AI Architecture

```text
User
 │
 ├── Typed Text
 │
 └── Voice
       │
       ▼
 Speech-to-Text
       │
       ▼
 ┌──────────────┐
 │  LangGraph   │
 │ Agent State  │
 └──────┬───────┘
        │
        ▼
      Router
        │
        ▼
   Vision Tool
   (Groq Vision)
        │
        ▼
    AI Triage
        │
   ┌────┼─────────┐
   │    │         │
   ▼    ▼         ▼
  LOW UNCERTAIN RED_FLAG
   │    │         │
   ▼    ▼         ▼
General Ask for  Escalation
Guidance More Info
   │    │         │
   └────┼─────────┘
        ▼
   Final Response
```

---

## What Makes It Agentic?

The original multimodal workflow followed a fixed sequence:

```text
Voice → Transcription → Vision Model → Response
```

This project extends that architecture with a **stateful LangGraph agent**.

The agent can:

- Maintain shared workflow state using `AgentState`
- Select an image or text analysis route
- Invoke a multimodal vision tool
- Observe the vision tool's output
- Reason over both the user's description and visual analysis
- Classify the situation into `LOW`, `UNCERTAIN`, or `RED_FLAG`
- Dynamically choose the next node using LangGraph conditional edges
- Request additional information when evidence is insufficient
- Route concerning cases to an escalation path
- Expose an execution trace showing the agent's decisions

---

## Agent Workflow

### 1. User Input

The user can either:

- Type a description of the skin concern
- Provide a voice description

A skin image is then uploaded for visual analysis.

### 2. Agent State

LangGraph maintains shared state across the workflow.

Example state:

```python
class AgentState(TypedDict, total=False):
    user_text: str
    image_path: str
    has_image: bool
    image_analysis: str
    risk_level: str
    route: str
    final_response: str
    trace: list[str]
```

This allows different nodes to read previous observations and add new information.

### 3. Router

The router determines which processing path should be used based on the available input.

For an uploaded image:

```text
Router → Image Analysis
```

### 4. Vision Tool

The image-analysis node calls the Groq multimodal model.

The tool receives:

```text
User description + Uploaded image
```

and returns visual observations and general informational guidance.

The result is stored in:

```python
state["image_analysis"]
```

### 5. AI Triage

The triage node reasons over:

```text
User Description
        +
Vision Tool Result
```

It returns structured output containing a risk classification and reason.

Example:

```json
{
  "risk_level": "UNCERTAIN",
  "reason": "The uploaded image does not clearly show the affected skin area."
}
```

Possible classifications:

```text
LOW
UNCERTAIN
RED_FLAG
```

### 6. Conditional Routing

LangGraph conditional edges determine which node executes next.

```text
LOW
 ↓
General informational guidance

UNCERTAIN
 ↓
Request additional information

RED_FLAG
 ↓
Escalation guidance
```

This conditional decision-making is the core agentic behavior of the system.

---

## Example Agent Execution Trace

The Gradio interface exposes the workflow execution so the routing can be inspected during a demo.

```text
✓ Input -> Typed text received
✓ Router -> Image analysis selected
✓ Vision Tool -> Analyzing uploaded skin image
✓ Vision Tool -> Analysis completed
✓ AI Triage -> LOW: No obvious urgent warning signs detected
✓ Safety -> Standard guidance path
```

For insufficient evidence:

```text
✓ Vision Tool -> Analysis completed
✓ AI Triage -> UNCERTAIN: Image is insufficient for reliable assessment
✓ Agent -> Additional information requested
```

For a concerning input:

```text
✓ Vision Tool -> Analysis completed
✓ AI Triage -> RED_FLAG: Potential urgent warning signs identified
✓ Safety -> Escalation path selected
```

---

## Three Agent Paths

### LOW

Used when the available information does not indicate obvious urgent warning signs.

```text
Vision Analysis
      ↓
AI Triage
      ↓
LOW
      ↓
General Skin-Care Guidance
```

### UNCERTAIN

Used when the agent does not have enough reliable information.

For example:

- Irrelevant image
- Unclear image
- Insufficient description
- Visual evidence does not match the user context

```text
Vision Analysis
      ↓
AI Triage
      ↓
UNCERTAIN
      ↓
Ask User for More Information
```

The system avoids guessing when the available evidence is insufficient.

### RED_FLAG

Used when the supplied information suggests potentially urgent symptoms.

```text
User Input
      ↓
Vision Analysis
      ↓
AI Triage
      ↓
RED_FLAG
      ↓
Escalation Path
```

Routine skin-care guidance is skipped and the user is directed toward appropriate professional evaluation.

---

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| LangGraph | Agent orchestration and conditional routing |
| Groq | LLM and multimodal vision inference |
| Qwen Vision | Image + text analysis |
| Gradio | Interactive web interface |
| Groq Whisper | Speech-to-text |
| Pillow | Image preprocessing |
| python-dotenv | Environment variable management |

---

## Project Structure

```text
ai-skin-specialist/
│
├── agent_graph.py
│   └── LangGraph state, nodes, routing and agent workflow
│
├── main.py
│   └── Gradio interface and LangGraph integration
│
├── brain_of_the_doctor_groq.py
│   └── Groq multimodal vision tool
│
├── voice_of_the_patient.py
│   └── Voice transcription
│
├── voice_of_the_doctor.py
│   └── Optional text-to-speech functionality
│
├── pyproject.toml
│   └── Python dependencies
│
├── sample.env
│   └── Environment-variable template
│
├── .gitignore
│   └── Protects API keys and local environment files
│
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd ai-skin-specialist
```

### 2. Create a virtual environment

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install .
```

On macOS, if PyAudio fails because PortAudio is missing:

```bash
brew install portaudio
```

Then:

```bash
pip install pyaudio
pip install .
```

---

## Environment Variables

Create a `.env` file in the project root:

```text
GROQ_API_KEY=your_groq_api_key_here
```

Never commit `.env` to GitHub.

The repository's `.gitignore` excludes:

```text
.env
venv/
__pycache__/
*.pyc
```

---

## Run the Application

```bash
python main.py
```

Gradio will display a local URL such as:

```text
http://127.0.0.1:7860
```

Open it in your browser.

---

## Using the Application

1. Type a skin concern or provide a voice description.
2. Upload a clear image of the affected area.
3. Click **Run Agentic Analysis**.
4. Review the skin-care guidance.
5. Inspect the **Agent Execution Trace** to see which nodes and routing path were executed.

---

## Demo Scenarios

### Normal / LOW Case

Example input:

```text
I have red itchy patches that started two days ago after using
a new face wash. There is no severe pain, facial swelling,
fever, or difficulty breathing.
```

Expected path:

```text
Vision Tool → AI Triage → LOW → Guidance
```

### UNCERTAIN Case

Upload an irrelevant or unclear image.

Expected path:

```text
Vision Tool → AI Triage → UNCERTAIN → Request More Information
```

### RED FLAG Case

Example input:

```text
My rash is spreading very rapidly. My face and lips are
swelling and I am having difficulty breathing.
```

Expected path:

```text
Vision Tool → AI Triage → RED_FLAG → Escalation
```

---

## Key Engineering Concepts Demonstrated

### Agentic AI

The system does not simply generate a response from one prompt. It observes tool results and selects subsequent actions.

### Tool Calling

The LangGraph image-analysis node invokes the external Groq multimodal capability and stores its result in agent state.

### Stateful Orchestration

`AgentState` allows observations and decisions to persist between graph nodes during execution.

### Conditional Edges

LangGraph routes execution dynamically according to the triage result.

### Multimodal AI

The system combines natural-language descriptions with visual information from uploaded images.

### Structured LLM Output

The triage component returns structured risk classification data that can be consumed programmatically by the graph.

### Observability

The application displays an Agent Execution Trace showing which tools, nodes, and routing decisions were executed.

### Safety-Aware Routing

Potentially concerning cases take an escalation path instead of receiving ordinary skin-care guidance.

---

## Example Interview Explanation

> I started with a multimodal skin-care assistant that followed a fixed voice-to-vision-to-response pipeline. I redesigned its orchestration layer using LangGraph. I introduced shared agent state, converted the vision capability into a callable tool, and built an AI triage node that reasons over both the user's description and the vision tool output. LangGraph conditional edges then dynamically route execution between normal guidance, clarification, and escalation paths. I also exposed the execution trace in the Gradio interface so the agent's decisions are observable during a live demo.

---

## Current Limitations

- The application provides informational guidance rather than medical diagnosis.
- Image-analysis quality depends on image clarity and the underlying multimodal model.
- The triage layer is an AI safety-routing mechanism and is not a validated clinical triage system.
- Video processing is not currently part of the active agent workflow.
- Text-to-speech is optional and may require an additional API configuration.

---

## Future Improvements

Potential extensions include:

- Conversation memory using LangGraph persistence
- Retrieval-Augmented Generation using trusted dermatology resources
- Human-in-the-loop review
- More sophisticated tool selection
- Agent evaluation datasets
- Confidence-based routing
- Persistent session history
- Additional multimodal tools

---

## Security

- API keys are stored in `.env`.
- `.env` is excluded through `.gitignore`.
- API keys should never be committed to GitHub.
- Images and voice recordings should be treated as potentially sensitive user data.

---

## Disclaimer

This project is an educational Agentic AI demonstration.

It provides general informational guidance only and does not diagnose medical conditions, prescribe medication, or replace evaluation by a qualified healthcare professional.