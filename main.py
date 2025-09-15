import base64
import os
import shutil

import requests
from fastapi import FastAPI, HTTPException
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel


A1111_URL = "http://127.0.0.1:7860"


@tool
def generate_image(prompt: str) -> str:
    """
    Use this tool to generate an image based on a detailed textual description.
    This tool calls a local Stable Diffusion API to create the image.
    The input should be a rich, descriptive prompt suitable for an image generation model.
    """
    print(f"--- Calling Image Generation Tool with prompt: '{prompt}' ---")

    payload = {
        "prompt": prompt,
        "negative_prompt": "blurry, ugly, deformed, watermark, text",
        "steps": 28,
        "width": 512,
        "height": 512,
        "cfg_scale": 7.0,
        "sampler_name": "DPM++ 2M Karras",
    }

    try:
        response = requests.post(url=f"{A1111_URL}/sdapi/v1/txt2img", json=payload)
        response.raise_for_status()
        r = response.json()

        if "images" in r and len(r["images"]) > 0:
            image_data = base64.b64decode(r["images"][0])

            # Use a hash of the prompt for a unique filename
            output_filename = f"img_{hash(prompt)}.png"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, "wb") as f:
                f.write(image_data)

            return f"Image successfully generated and saved to {output_path}"
        else:
            return "Error: API response did not contain an image."

    except Exception as e:
        print(f"Error calling AUTOMATIC1111 API: {e}")
        return f"Error: Failed to generate image. The service may be down. Details: {str(e)}"


# 2. Initialize the LLM
llm = ChatOllama(model="llama3.1:8b", temperature=0)

# 3. Define the list of tools the agent can use
tools = [generate_image]

# 4. Create the Agent
# The create_react_agent function will use a default prompt suitable for ReAct agents.
agent = create_react_agent(llm, tools)

# 5. Create the Agent Executor
# The prompt is part of the agent, so we don't need to pass it here.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


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


@app.post("/agent/generate")
async def agent_generate(request: GenerationRequest):
    """
    Accepts a prompt and uses the LangChain agent to generate a response,
    which may include generating an image.
    """
    print(f"--- Calling Agent with prompt: '{request.prompt}' ---")
    # The agent expects a dictionary with an 'input' key for the prompt
    # and a 'tools' key for the list of tools.
    # The 'tools' key is implicitly handled by create_react_agent,
    # but the 'input' key is explicitly needed for the prompt.
    response = await agent_executor.ainvoke({"input": request.prompt})
    return {"response": response}
