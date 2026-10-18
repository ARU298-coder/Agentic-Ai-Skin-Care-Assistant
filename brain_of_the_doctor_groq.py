import base64
import os
from io import BytesIO

from dotenv import load_dotenv
from groq import Groq
from PIL import Image


load_dotenv()


def encode_image_for_groq(filepath):
    image = Image.open(filepath)
    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    image.convert("RGB").save(
        buffer,
        format="JPEG",
        quality=75
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")


def brain_of_the_doctor(
    patient_text,
    image_filepath=None,
    video_filepath=None
):
    groq_api_key = os.environ.get("GROQ_API_KEY")

    if not groq_api_key:
        raise ValueError(
            "Missing GROQ_API_KEY in .env or environment"
        )

    if not image_filepath:
        raise ValueError(
            "Please upload a skin image."
        )

    # Convert image to base64
    image_data = encode_image_for_groq(image_filepath)

    prompt = (
        "You are a careful AI skin-care assistant. "
        "Analyze the visible skin-related features in the image "
        "together with the user's description. "
        "Provide general informational guidance only and do not "
        "claim to make a medical diagnosis. "
        "Keep the response concise, around two or three sentences. "
        "Mention uncertainty when the image is insufficient.\n\n"
        f"User description: {patient_text}"
    )

    if video_filepath:
        prompt += (
            "\nA video was also provided, but this analysis "
            "uses the uploaded image as the visual input."
        )

    client = Groq(api_key=groq_api_key)

    response = client.chat.completions.create(
        model=os.environ.get(
            "GROQ_MODEL",
            "qwen/qwen3.8-27b"
        ),
        max_completion_tokens=500,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful skin-care information "
                    "assistant. Do not provide a definitive diagnosis."
                ),
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/jpeg;base64,"
                                + image_data
                            )
                        }
                    }
                ]
            }
        ]
    )

    return response.choices[0].message.content