import base64
import operator
import os
from typing import Annotated

from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

load_dotenv(find_dotenv())
openai_41_base_model = init_chat_model(
    model="gpt-4.1-2025-04-14",
    api_key=os.getenv("OPENAI_API_KEY"),
)


class Picture(TypedDict):
    pic_path: str


class Video(TypedDict, total=False):
    name: str
    pic: Picture
    prompt: str
    download_url: str
    file_path: str


class OverallState(TypedDict):
    file_to_process: Annotated[Video, operator.add]


def collect_file_to_process(state: OverallState) -> OverallState:
    pic_input_folder = "pic-input"
    supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    files_to_process = []

    if not os.path.exists(pic_input_folder):
        print(f"Warning: {pic_input_folder} folder does not exist")
        return {"file_to_process": files_to_process}

    for filename in os.listdir(pic_input_folder):
        file_path = os.path.join(pic_input_folder, filename)

        if os.path.isdir(file_path):
            continue

        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in supported_formats:
            continue

        picture = Picture(pic_path=file_path)

        video = Video(pic=picture)

        files_to_process.append(video)
        print(f"Added {filename} to processing queue")

    print(f"Found {len(files_to_process)} files to process")
    return {"file_to_process": files_to_process}


def generate_prompt(state: Video) -> Video:
    class ExtractNameAndPromptSchema(BaseModel):
        name: str = Field(description="The name of the video")
        prompt: str = Field(description="The prompt for the video")

    extraction_model = openai_41_base_model.with_structured_output(
        ExtractNameAndPromptSchema
    )

    # Get the image path from the state
    pic_path = state["pic"]["pic_path"]

    # Read and encode the image in base64
    try:
        with open(pic_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("utf-8")

        # Get the image file extension to determine MIME type
        file_ext = os.path.splitext(pic_path)[1].lower()
        mime_type_map = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".tiff": "image/tiff",
            ".webp": "image/webp",
        }
        mime_type = mime_type_map.get(file_ext, "image/jpeg")

        # Create the message with correct LangChain format
        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """Analyze this image and provide:
                    1. A descriptive name for a video that could be generated from this image
                    2. A detailed prompt that describes the scene, objects, actions, mood, and visual elements that would be good for video generation

                    The name should be concise and descriptive.
                    The prompt should be detailed and include information about movement, lighting, atmosphere, and any dynamic elements that would translate well to video.""",
                },
                {
                    "type": "image",
                    "source_type": "base64",
                    "data": image_data,
                    "mime_type": mime_type,
                },
            ],
        }

        # Get the structured response from the LLM
        response = extraction_model.invoke([message])

        # Update the state with extracted information
        updated_state = state.copy()
        updated_state["name"] = response.name
        updated_state["prompt"] = response.prompt

        print(f"Generated name: {response.name}")
        print(f"Generated prompt: {response.prompt}")

        return updated_state

    except Exception as e:
        print(f"Error processing image {pic_path}: {e}")
        # Return state with default values in case of error
        updated_state = state.copy()
        updated_state["name"] = f"video_from_{os.path.basename(pic_path)}"
        updated_state["prompt"] = "A scene based on the uploaded image"
        return updated_state
