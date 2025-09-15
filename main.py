from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import shutil

app = FastAPI(
    title="Agentic Image Generation API",
    description="An API for generating images using a local agentic system.",
    version="0.1.0",
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Agentic API"}


OUTPUT_DIR = "final_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class GenerationRequest(BaseModel):
    prompt: str


class GenerationResponse(BaseModel):
    message: str
    image_path: str


# --- Mocked Functions ---
def decide_if_image_is_needed(prompt: str) -> bool:
    print(f"[MOCK] Deciding if image is needed for: '{prompt}'")
    # For the mock, we assume an image is always needed if "image" or "create" is in the prompt
    return "image" in prompt.lower() or "create" in prompt.lower()


def generate_image_prompt(prompt: str) -> str:
    print(f"[MOCK] Generating a specialized image prompt from: '{prompt}'")
    # Mock behavior: just add a suffix
    return f"photorealistic, cinematic style, {prompt}"


def call_image_generator_api(image_prompt: str) -> str:
    print(f"[MOCK] Calling image generator with prompt: '{image_prompt}'")
    # Mock behavior: copy a placeholder image instead of calling the API
    placeholder_image = "placeholder.png"
    # Create a dummy placeholder if it doesn't exist
    if not os.path.exists(placeholder_image):
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="red")
        img.save(placeholder_image)

    output_filename = f"{hash(image_prompt)}.png"
    destination_path = os.path.join(OUTPUT_DIR, output_filename)
    shutil.copy(placeholder_image, destination_path)
    return destination_path


@app.post("/generate-mocked", response_model=GenerationResponse)
def generate_mocked_image(request: GenerationRequest):
    if not decide_if_image_is_needed(request.prompt):
        raise HTTPException(
            status_code=400,
            detail="The prompt did not seem to require image generation.",
        )

    image_prompt = generate_image_prompt(request.prompt)
    image_path = call_image_generator_api(image_prompt)

    return GenerationResponse(
        message="Image generation process (mocked) completed.", image_path=image_path
    )
