
import os
import mimetypes

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def describe_command(image_path: str, query:str):

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"
    
    with open(image_path, "rb") as f:
        img = f.read()

    prompt = f"""
Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""
    
    parts = [
        prompt,
        types.Part.from_bytes(data=img, mime_type=mime),
        query.strip(),
    ]
    response = client.models.generate_content(model=model, contents=parts)

    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


