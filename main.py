from pathlib import Path

import gradio as gr

from agent_graph import agent
from voice_of_the_patient import transcribe_patient_voice


APP_TITLE = "Agentic AI Skin-Care Assistant"


CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@100..900&display=swap');

:root {
    --ais-bg: #f7f9fb;
    --ais-surface: #ffffff;
    --ais-surface-low: #f2f4f6;
    --ais-border: #c6c6cd;
    --ais-text: #191c1e;
    --ais-muted: #45464d;
    --ais-primary: #0051d5;
    --ais-primary-active: #316bf3;
    --ais-radius: 16px;
}

.gradio-container {
    background: var(--ais-bg) !important;
    color: var(--ais-text) !important;
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", system-ui, sans-serif !important;
}

.ais-shell {
    max-width: 1280px;
    margin: 0 auto;
    padding: 28px 40px 40px;
}

.ais-topbar {
    align-items: center;
    background: var(--ais-surface);
    border: 1px solid var(--ais-border);
    border-radius: 20px;
    display: flex;
    justify-content: space-between;
    margin-bottom: 32px;
    padding: 20px 24px;
}

.ais-brand h1 {
    color: var(--ais-text);
    font-size: 24px;
    font-weight: 650;
    margin: 0;
}

.ais-brand p {
    color: var(--ais-muted);
    font-size: 12px;
    font-weight: 600;
    margin: 4px 0 0;
}

.ais-security {
    color: var(--ais-muted);
    font-size: 12px;
    font-weight: 600;
}

.ais-grid {
    align-items: start;
    display: grid;
    gap: 24px;
    grid-template-columns: minmax(0, 5fr) minmax(0, 7fr);
}

.ais-section-title {
    margin-bottom: 14px;
}

.ais-section-title h2 {
    color: var(--ais-text);
    font-size: 24px;
    font-weight: 650;
    margin: 0;
}

.ais-card {
    background: var(--ais-surface);
    border: 1px solid var(--ais-border);
    border-radius: var(--ais-radius);
    box-shadow: 0 12px 34px rgba(19, 27, 46, 0.06);
    padding: 20px;
}

.ais-input-card {
    display: flex;
    flex-direction: column;
    gap: 16px;
}

.ais-media-row {
    display: grid;
    gap: 16px;
    grid-template-columns: repeat(2, minmax(0, 1fr));
}

.ais-submit-wrap .gr-button-primary {
    background: var(--ais-primary-active) !important;
    border: 0 !important;
    border-radius: 16px !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 650 !important;
    min-height: 58px !important;
}

.ais-submit-wrap .gr-button-primary:hover {
    background: var(--ais-primary) !important;
}

.ais-note {
    background: #eef2ff;
    border-radius: 12px;
    color: #003ea8;
    font-size: 12px;
    font-weight: 600;
    padding: 14px 16px;
}

.ais-response-card {
    min-height: 590px;
}

.ais-output-stack {
    display: flex;
    flex-direction: column;
    gap: 18px;
}

.ais-output-stack .gradio-textbox textarea {
    background: var(--ais-surface-low) !important;
    border: 0 !important;
    color: var(--ais-text) !important;
    font-size: 15px !important;
    line-height: 23px !important;
}

.ais-transcript textarea {
    color: var(--ais-muted) !important;
    font-style: italic;
}

.ais-trace textarea {
    font-family: monospace !important;
    font-size: 13px !important;
}

.ais-footer {
    background: var(--ais-surface-low);
    border: 1px solid var(--ais-border);
    border-radius: 16px;
    color: var(--ais-muted);
    font-size: 12px;
    margin-top: 32px;
    padding: 16px 20px;
}

@media (max-width: 900px) {
    .ais-shell {
        padding: 20px;
    }

    .ais-grid,
    .ais-media-row {
        grid-template-columns: 1fr;
    }
}
"""


# ---------------------------------------------------------
# MAIN PROCESSING FUNCTION
# ---------------------------------------------------------

def process_inputs(
    text_input_value,
    audio_filepath,
    image_filepath,
    video_filepath
):

    # -----------------------------------------------------
    # 1. GET USER DESCRIPTION
    # -----------------------------------------------------

    if text_input_value and text_input_value.strip():

        patient_text = text_input_value.strip()
        input_method = "Typed text"

    elif audio_filepath:

        patient_text = transcribe_patient_voice(audio_filepath)
        input_method = "Voice transcription"

    else:

        raise gr.Error(
            "Please type your skin concern or record/upload "
            "a voice description."
        )

    # -----------------------------------------------------
    # 2. REQUIRE IMAGE
    # -----------------------------------------------------

    if not image_filepath:

        raise gr.Error(
            "Please upload a clear image of the affected skin area."
        )

    # -----------------------------------------------------
    # 3. RUN LANGGRAPH AGENT
    # -----------------------------------------------------

    result = agent.invoke(
        {
            "user_text": patient_text,
            "image_path": image_filepath,
            "has_image": True,
            "trace": [
                f"Input -> {input_method} received"
            ]
        }
    )

    # -----------------------------------------------------
    # 4. GET FINAL AGENT RESPONSE
    # -----------------------------------------------------

    doctor_text = result.get(
        "final_response",
        "The agent could not generate a response."
    )

    # -----------------------------------------------------
    # 5. FORMAT AGENT EXECUTION TRACE
    # -----------------------------------------------------

    trace_steps = result.get("trace", [])

    agent_trace = "\n".join(
        f"✓ {step}"
        for step in trace_steps
    )

    # -----------------------------------------------------
    # 6. AUDIO TEMPORARILY DISABLED
    # -----------------------------------------------------
    # Deepgram TTS can be added back after
    # DEEPGRAM_API_KEY is configured.

    doctor_audio = None

    return (
        patient_text,
        doctor_text,
        agent_trace,
        doctor_audio
    )


# ---------------------------------------------------------
# GRADIO UI
# ---------------------------------------------------------

with gr.Blocks(title=APP_TITLE) as iface:

    with gr.Column(elem_classes="ais-shell"):

        # -------------------------------------------------
        # HEADER
        # -------------------------------------------------

        gr.HTML(
            """
            <header class="ais-topbar">

                <div class="ais-brand">
                    <h1>Agentic AI Skin-Care Assistant</h1>

                    <p>
                    MULTIMODAL LANGGRAPH AGENT WITH
                    VISION ANALYSIS AND SAFETY TRIAGE
                    </p>
                </div>

                <div class="ais-security">
                    Safety-aware Agentic AI
                </div>

            </header>
            """
        )

        # -------------------------------------------------
        # MAIN GRID
        # -------------------------------------------------

        with gr.Row(elem_classes="ais-grid"):

            # =============================================
            # LEFT SIDE - USER INPUT
            # =============================================

            with gr.Column(scale=5):

                gr.HTML(
                    """
                    <div class="ais-section-title">
                        <h2>Patient Input</h2>
                    </div>
                    """
                )

                with gr.Column(
                    elem_classes="ais-card ais-input-card"
                ):

                    # -------------------------------------
                    # TEXT INPUT
                    # -------------------------------------

                    text_input = gr.Textbox(
                        label="Type Your Skin Concern",
                        placeholder=(
                            "Example: I have red itchy patches "
                            "that started two days ago after "
                            "using a new face wash."
                        ),
                        lines=4,
                    )

                    # -------------------------------------
                    # VOICE INPUT
                    # -------------------------------------

                    audio_input = gr.Audio(
                        sources=[
                            "microphone",
                            "upload"
                        ],
                        type="filepath",
                        label="Or Describe It By Voice",
                    )

                    # -------------------------------------
                    # IMAGE / VIDEO
                    # -------------------------------------

                    with gr.Row(
                        elem_classes="ais-media-row"
                    ):

                        image_input = gr.Image(
                            type="filepath",
                            label="Skin Image",
                            height=280,
                        )

                        video_input = gr.Video(
                            label="Skin Video",
                            height=280,
                        )

                    # -------------------------------------
                    # ANALYZE BUTTON
                    # -------------------------------------

                    with gr.Column(
                        elem_classes="ais-submit-wrap"
                    ):

                        analyze_button = gr.Button(
                            "Run Agentic Analysis",
                            variant="primary",
                            size="lg",
                        )

                    gr.HTML(
                        """
                        <div class="ais-note">

                            The AI agent analyzes the image,
                            evaluates the available context,
                            and dynamically routes the request
                            through LOW, UNCERTAIN, or RED FLAG
                            safety paths.

                        </div>
                        """
                    )

            # =============================================
            # RIGHT SIDE - AGENT OUTPUT
            # =============================================

            with gr.Column(scale=7):

                gr.HTML(
                    """
                    <div class="ais-section-title">
                        <h2>Agent Response</h2>
                    </div>
                    """
                )

                with gr.Column(
                    elem_classes="ais-card ais-response-card"
                ):

                    with gr.Column(
                        elem_classes="ais-output-stack"
                    ):

                        # ---------------------------------
                        # USER TEXT
                        # ---------------------------------

                        transcript_output = gr.Textbox(
                            label="Patient Description",
                            lines=4,
                            interactive=False,
                            elem_classes="ais-transcript",
                        )

                        # ---------------------------------
                        # FINAL GUIDANCE
                        # ---------------------------------

                        response_output = gr.Textbox(
                            label="Skin-Care Guidance",
                            lines=9,
                            interactive=False,
                        )

                        # ---------------------------------
                        # AGENT TRACE
                        # ---------------------------------

                        agent_trace_output = gr.Textbox(
                            label="Agent Execution Trace",
                            lines=9,
                            interactive=False,
                            elem_classes="ais-trace",
                        )

                        # ---------------------------------
                        # AUDIO OUTPUT
                        # ---------------------------------

                        audio_output = gr.Audio(
                            label="Voice Response",
                            type="filepath",
                            autoplay=False,
                        )

                gr.HTML(
                    """
                    <div
                        class="ais-security"
                        style="
                            justify-content:center;
                            margin-top:16px;
                        "
                    >
                        AI guidance is informational and
                        is not a medical diagnosis.
                    </div>
                    """
                )

        # -------------------------------------------------
        # FOOTER
        # -------------------------------------------------

        gr.HTML(
            """
            <footer class="ais-footer">

                <strong>
                    Agentic AI Skin-Care Assistant
                </strong>

                <br>

                For urgent or serious symptoms,
                seek appropriate professional medical care.

            </footer>
            """
        )

    # -----------------------------------------------------
    # BUTTON EVENT
    # -----------------------------------------------------

    analyze_button.click(
        fn=process_inputs,

        inputs=[
            text_input,
            audio_input,
            image_input,
            video_input,
        ],

        outputs=[
            transcript_output,
            response_output,
            agent_trace_output,
            audio_output,
        ],
    )


# ---------------------------------------------------------
# START APPLICATION
# ---------------------------------------------------------

if __name__ == "__main__":

    iface.launch(
        debug=True,
        css=CSS,
        theme=gr.themes.Base(),
    )