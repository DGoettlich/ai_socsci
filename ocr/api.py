"""
SETUP: Define OPENAI_API_KEY in your environment variables before running this script.
"""

from openai import OpenAI
from pathlib import Path
from dotenv import load_dotenv
import base64

load_dotenv(override=True)

PAGES_ROOT = Path("data/ocr/pages")
OUT_ROOT = Path("data/ocr/transcriptions")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# ___PROMPT___

prompt = """
transcribe the text on the provided document page image as accurately as possible.
"""

pages = sorted(PAGES_ROOT.glob("*.png"))

client = OpenAI()

for page in pages:
    base64_image = encode_image(page)

    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )

    out_path = OUT_ROOT / f"{page.stem}.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(response.output_text)
