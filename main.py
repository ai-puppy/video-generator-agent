import base64
import operator
import os
from typing import Annotated
from urllib.parse import urlparse

import fal_client
import requests
from dotenv import find_dotenv, load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    VideoFileClip,
    concatenate_videoclips,
)
from moviepy.audio.fx import MultiplyVolume
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
    files_to_process: list[Video]  # Input: videos without prompts
    processed_videos: Annotated[
        list[Video], operator.add
    ]  # Output: videos with prompts
    final_videos: list[Video]  # Final: videos with consistent prompts
    completed_videos: Annotated[
        list[Video], operator.add
    ]  # Completed: videos with generated files
    combined_video_path: str  # Path to the final combined video
    music_file_path: str  # Path to the generated music file
    final_video_with_music_path: str  # Path to the final video with music


def collect_file_to_process(state: OverallState) -> OverallState:
    pic_input_folder = "pic-input"
    supported_formats = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    files_to_process = []

    if not os.path.exists(pic_input_folder):
        print(f"Warning: {pic_input_folder} folder does not exist")
        return {"files_to_process": files_to_process}

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
    return {"files_to_process": files_to_process}


def generate_prompt(state: Video) -> OverallState:
    class ExtractNameAndPromptSchema(BaseModel):
        name: str = Field(description="The name of the video")
        prompt: str = Field(description="The prompt for the video")

    extraction_model = openai_41_base_model.with_structured_output(
        ExtractNameAndPromptSchema
    )

    pic_path = state["pic"]["pic_path"]

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

        # Return in the format expected by the reducer (wrapped in a list)
        return {"processed_videos": [updated_state]}

    except Exception as e:
        print(f"Error processing image {pic_path}: {e}")
        # Return state with default values in case of error
        updated_state = state.copy()
        updated_state["name"] = f"video_from_{os.path.basename(pic_path)}"
        updated_state["prompt"] = "A scene based on the uploaded image"
        return {"processed_videos": [updated_state]}


def check_prompt_consistency(state: OverallState) -> OverallState:
    """
    Analyze all generated prompts for style consistency and modify them if needed
    to ensure they work well together as a connected video sequence.
    """

    class ConsistentPrompt(BaseModel):
        name: str = Field(description="The name of the video")
        prompt: str = Field(description="The adjusted prompt for the video")

    class ConsistentPromptsSchema(BaseModel):
        consistent_prompts: list[ConsistentPrompt] = Field(
            description="List of videos with adjusted names and prompts for consistency"
        )
        style_notes: str = Field(
            description="Notes about the consistent style applied to all prompts"
        )

    consistency_model = openai_41_base_model.with_structured_output(
        ConsistentPromptsSchema
    )

    # Extract all current prompts and names from processed videos
    current_videos = state["processed_videos"]

    if not current_videos:
        print("No videos to check for consistency")
        return state

    # Prepare the prompt analysis text
    prompt_analysis = "Current video prompts to analyze for consistency:\n\n"
    for i, video in enumerate(current_videos, 1):
        name = video.get("name", f"video_{i}")
        prompt = video.get("prompt", "No prompt generated")
        prompt_analysis += f"{i}. Name: {name}\n   Prompt: {prompt}\n\n"

    message = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"""
                {prompt_analysis}

                These prompts will be used to generate videos that will be connected together into a cohesive sequence.

                Please analyze these prompts and ensure they have:
                1. Consistent visual style (lighting, color palette, mood)
                2. Consistent camera movement and perspective
                3. Consistent pacing and energy level
                4. Smooth transitions between scenes

                Modify the prompts as needed to create a cohesive video sequence while preserving the core content of each scene.
                Return the adjusted prompts that will work well together.
                """,
            }
        ],
    }

    try:
        response = consistency_model.invoke([message])

        # Update the state with consistent prompts
        updated_videos = []
        for i, video in enumerate(current_videos):
            updated_video = video.copy()
            if i < len(response.consistent_prompts):
                consistent_prompt = response.consistent_prompts[i]
                updated_video["name"] = consistent_prompt.name
                updated_video["prompt"] = consistent_prompt.prompt
            updated_videos.append(updated_video)

        print(f"Style consistency applied: {response.style_notes}")
        print(f"Updated {len(updated_videos)} video prompts for consistency")

        return {"final_videos": updated_videos}

    except Exception as e:
        print(f"Error checking prompt consistency: {e}")
        return state


def continue_to_prompt_generation(state: OverallState):
    """
    Map function to send each collected file to prompt generation
    """
    return [Send("generate_prompt", video) for video in state["files_to_process"]]


def continue_to_video_generation(state: OverallState):
    """
    Map function to send each final video to video generation
    """
    return [Send("generate_video", video) for video in state["final_videos"]]


def generate_video(state: Video) -> OverallState:
    """
    Generate a video from an image and prompt using fal_client
    """
    pic_path = state["pic"]["pic_path"]
    prompt = state.get("prompt", "A beautiful scene")
    name = state.get("name", "generated_video")

    print(f"Starting video generation for: {name}")

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"[{name}] {log['message']}")

    def download_video_to_output(video_url, filename=None):
        """Download video from URL to video-output folder"""
        try:
            print(f"Downloading video from: {video_url}")

            # Ensure output directory exists
            output_dir = "video-output"
            os.makedirs(output_dir, exist_ok=True)

            # Get filename from URL if not provided
            if not filename:
                parsed_url = urlparse(video_url)
                base_filename = os.path.basename(parsed_url.path)
                if not base_filename or not base_filename.endswith(".mp4"):
                    base_filename = "generated_video.mp4"
                filename = base_filename

            # Create full file path
            file_path = os.path.join(output_dir, filename)

            # Download the video
            response = requests.get(video_url, stream=True)
            response.raise_for_status()

            # Save to file
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Video downloaded successfully as: {file_path}")
            return file_path

        except Exception as e:
            print(f"Error downloading video: {e}")
            return None

    try:
        # Upload the image file
        print(f"Uploading image: {pic_path}")
        image_url = fal_client.upload_file(pic_path)
        print(f"Image uploaded successfully: {image_url}")

        # Generate the video
        print(f"Generating video with prompt: {prompt[:100]}...")
        result = fal_client.subscribe(
            "fal-ai/kling-video/v2.1/standard/image-to-video",
            arguments={"prompt": prompt, "image_url": image_url},
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        print(f"Video generation result: {result}")

        # Download the video if generation was successful
        if result and "video" in result and "url" in result["video"]:
            video_url = result["video"]["url"]

            # Create unique filename based on the video name
            safe_name = "".join(c for c in name if c.isalnum() or c in " -_")
            safe_name = safe_name.replace(" ", "_")  # Replace spaces with underscores
            unique_filename = f"{safe_name}.mp4"

            # Download the video
            downloaded_file = download_video_to_output(video_url, unique_filename)

            if downloaded_file:
                file_size = os.path.getsize(downloaded_file)
                print(f"Downloaded file size: {file_size:,} bytes")

                # Update the video state with download information
                updated_state = state.copy()
                updated_state["download_url"] = video_url
                updated_state["file_path"] = downloaded_file

                return {"completed_videos": [updated_state]}
            else:
                print("Failed to download video")
                return {"completed_videos": [state]}
        else:
            print("No video URL found in the result")
            return {"completed_videos": [state]}

    except Exception as e:
        print(f"Error generating video for {name}: {e}")
        return {"completed_videos": [state]}


def combine_videos(state: OverallState) -> OverallState:
    """
    Combine all generated videos into one final video
    """
    completed_videos = state["completed_videos"]

    if not completed_videos:
        print("No completed videos to combine")
        return state

    # Filter videos that have valid file paths
    valid_videos = [
        video
        for video in completed_videos
        if video.get("file_path") and os.path.exists(video["file_path"])
    ]

    if not valid_videos:
        print("No valid video files found to combine")
        return state

    try:
        print(f"Combining {len(valid_videos)} videos...")

        # Sort videos by their original image filename for consistent order
        valid_videos.sort(key=lambda v: os.path.basename(v["pic"]["pic_path"]))

        # Load video clips
        clips = []
        for video in valid_videos:
            file_path = video["file_path"]
            print(f"Loading video: {file_path}")

            try:
                clip = VideoFileClip(file_path)
                clips.append(clip)
                print(f"  Duration: {clip.duration:.2f}s, Size: {clip.size}")
            except Exception as e:
                print(f"  Error loading {file_path}: {e}")
                continue

        if not clips:
            print("No valid clips could be loaded")
            return state

        # Concatenate all clips
        print("Concatenating videos...")
        final_clip = concatenate_videoclips(clips, method="compose")

        # Create output filename
        output_dir = "video-output"
        os.makedirs(output_dir, exist_ok=True)
        combined_filename = "combined_video.mp4"
        combined_path = os.path.join(output_dir, combined_filename)

        # Write the final video
        print(f"Writing combined video to: {combined_path}")
        final_clip.write_videofile(
            combined_path,
            codec="libx264",
            audio_codec="aac",
            logger=None,  # Suppress moviepy logs
        )

        # Close all clips to free memory
        final_clip.close()
        for clip in clips:
            clip.close()

        # Get file info
        file_size = os.path.getsize(combined_path)
        print("Combined video created successfully!")
        print(f"  Path: {combined_path}")
        print(f"  Size: {file_size:,} bytes")
        print(f"  Duration: {final_clip.duration:.2f}s")

        return {"combined_video_path": combined_path}

    except Exception as e:
        print(f"Error combining videos: {e}")
        return state


def generate_music(state: OverallState) -> OverallState:
    """
    Generate background music based on the number of video clips.
    Duration calculation: 5 seconds * number_of_clips + 1 second
    """
    completed_videos = state.get("completed_videos", [])

    # Calculate music duration based on number of clips
    num_clips = len(completed_videos)
    if num_clips == 0:
        print("No videos found for music generation")
        return state

    music_duration = (num_clips * 5) + 1  # 5 seconds per clip + 1 extra second
    print(f"Generating music for {num_clips} clips, duration: {music_duration} seconds")

    # Generate a custom music prompt based on video content
    def generate_music_prompt() -> str:
        """Generate a music prompt based on the video content using LLM"""

        class MusicPromptSchema(BaseModel):
            music_prompt: str = Field(description="A detailed music generation prompt")
            music_style_notes: str = Field(
                description="Notes about the chosen music style and reasoning"
            )
            music_name: str = Field(
                description="A descriptive name for this background music"
            )

        music_prompt_model = openai_41_base_model.with_structured_output(
            MusicPromptSchema
        )

        # Gather all video information
        video_content_analysis = "Video content to create music for:\n\n"
        for i, video in enumerate(completed_videos, 1):
            name = video.get("name", f"video_{i}")
            prompt = video.get("prompt", "No prompt available")
            video_content_analysis += f"{i}. Video Name: {name}\n"
            video_content_analysis += f"   Video Description: {prompt}\n\n"

        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    {video_content_analysis}

                    Based on the video content above, create a music generation prompt that will produce touching and heartwarming background music that complements these videos.

                    Requirements:
                    1. The music should be emotionally touching and heartwarming
                    2. It should complement the overall mood and theme of the videos
                    3. Include professional music details like:
                       - Key signature
                       - Tempo (BPM)
                       - Instruments
                       - Musical style/genre
                       - Emotional tone
                    4. The music should work well as background music for a video montage
                    5. Consider the narrative flow between the different video scenes
                    6. Make it suitable for the duration of {music_duration} seconds

                    Also provide a descriptive name for this background music that captures the essence of the video content.

                    Create a detailed music prompt that will generate beautiful, professional background music.
                    """,
                }
            ],
        }

        try:
            response = music_prompt_model.invoke([message])
            print(f"Generated music name: {response.music_name}")
            print(f"Generated music style: {response.music_style_notes}")
            return response.music_prompt, response.music_name
        except Exception as e:
            print(f"Error generating custom music prompt: {e}")
            # Fallback to a generic heartwarming prompt
            return (
                "Gentle, heartwarming acoustic music with soft piano, warm strings, and subtle percussion. Key: C Major, Tempo: 75 BPM. Emotional and touching melody perfect for life moments and memories.",
                "Heartwarming_Journey",
            )

    # Generate custom music prompt and name
    custom_music_prompt, music_name = generate_music_prompt()
    print(f"Using custom music prompt: {custom_music_prompt}")

    def on_queue_update(update):
        if isinstance(update, fal_client.InProgress):
            for log in update.logs:
                print(f"[Music Generation] {log['message']}")

    def download_music_file(result, download_dir="music-output"):
        """Download the generated music file"""
        try:
            # Create download directory if it doesn't exist
            os.makedirs(download_dir, exist_ok=True)

            # Extract audio file information
            audio_file = result.get("audio_file")
            if not audio_file:
                raise ValueError("No audio_file found in result")

            url = audio_file.get("url")
            original_file_name = audio_file.get("file_name", "generated_music.wav")

            # Create custom filename based on generated name
            # Clean the music name to be filesystem-safe
            safe_music_name = "".join(
                c for c in music_name if c.isalnum() or c in " -_"
            )
            safe_music_name = safe_music_name.replace(" ", "_")

            # Get file extension from original filename
            file_ext = os.path.splitext(original_file_name)[1] or ".wav"
            custom_file_name = f"{safe_music_name}{file_ext}"

            if not url:
                raise ValueError("No URL found in audio_file")

            # Download the file
            print(f"Downloading music file from: {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()

            # Save the file with custom name
            file_path = os.path.join(download_dir, custom_file_name)
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"Music file downloaded successfully: {file_path}")
            print(f"File size: {audio_file.get('file_size', 'Unknown')} bytes")

            return file_path
        except Exception as e:
            print(f"Error downloading music: {e}")
            return None

    try:
        # Generate music using fal_client with custom prompt
        result = fal_client.subscribe(
            "CassetteAI/music-generator",
            arguments={"prompt": custom_music_prompt, "duration": music_duration},
            with_logs=True,
            on_queue_update=on_queue_update,
        )

        print(f"Music generation result: {result}")

        # Download the generated music file
        music_file_path = download_music_file(result)

        if music_file_path:
            return {"music_file_path": music_file_path}
        else:
            print("Failed to download music file")
            return state

    except Exception as e:
        print(f"Error generating music: {e}")
        return state


def add_music_to_video(state: OverallState) -> OverallState:
    """
    Add the generated background music to the combined video.
    """
    combined_video_path = state.get("combined_video_path")
    music_file_path = state.get("music_file_path")
    completed_videos = state.get("completed_videos", [])

    if not combined_video_path or not os.path.exists(combined_video_path):
        print("No combined video found to add music to")
        return state

    if not music_file_path or not os.path.exists(music_file_path):
        print("No music file found to add to video")
        return state

    # Generate a custom name for the final video
    def generate_video_name() -> str:
        """Generate a meaningful name for the final video based on content"""

        class VideoNameSchema(BaseModel):
            video_name: str = Field(
                description="A descriptive name for the final video"
            )
            name_reasoning: str = Field(
                description="Brief explanation of why this name was chosen"
            )

        video_name_model = openai_41_base_model.with_structured_output(VideoNameSchema)

        # Gather all video information
        video_content_summary = "Content of the video montage:\n\n"
        for i, video in enumerate(completed_videos, 1):
            name = video.get("name", f"video_{i}")
            prompt = video.get("prompt", "No prompt available")
            video_content_summary += f"{i}. {name}: {prompt}\n"

        message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"""
                    {video_content_summary}

                    Based on the video content above, create a descriptive and meaningful name for the final video montage.

                    Requirements:
                    1. The name should capture the overall theme or story of the video collection
                    2. It should be suitable as a filename (no special characters that would cause issues)
                    3. It should be concise but descriptive (2-5 words)
                    4. It should feel professional and engaging
                    5. Consider the emotional journey or narrative flow

                    Examples of good names: "Sunset_Memories", "Family_Journey", "Urban_Adventures", "Nature_Escape", "Life_Moments"

                    Provide a name that would make someone want to watch this video.
                    """,
                }
            ],
        }

        try:
            response = video_name_model.invoke([message])
            print(f"Generated video name: {response.video_name}")
            print(f"Name reasoning: {response.name_reasoning}")

            # Clean the name to be filesystem-safe
            safe_name = "".join(
                c for c in response.video_name if c.isalnum() or c in " -_"
            )
            safe_name = safe_name.replace(" ", "_")
            return safe_name
        except Exception as e:
            print(f"Error generating custom video name: {e}")
            # Fallback to a generic but meaningful name
            return "Video_Story_with_Music"

    try:
        print(f"Adding music to video: {combined_video_path}")
        print(f"Using music file: {music_file_path}")

        # Generate custom video name
        custom_video_name = generate_video_name()

        # Load video and audio
        video = VideoFileClip(combined_video_path)
        background_music = AudioFileClip(music_file_path)

        # Adjust audio volume (30% of original volume)
        background_music = background_music.with_effects([MultiplyVolume(0.3)])

        # Handle audio duration vs video duration
        if background_music.duration < video.duration:
            print(
                f"Music ({background_music.duration:.2f}s) is shorter than video ({video.duration:.2f}s). Looping music."
            )
            # Loop the audio to match video duration
            background_music = background_music.loop(duration=video.duration)
        elif background_music.duration > video.duration:
            print(
                f"Music ({background_music.duration:.2f}s) is longer than video ({video.duration:.2f}s). Trimming music."
            )
            # Trim audio to match video duration
            background_music = background_music.subclipped(0, video.duration)

        # Combine original video audio with background music (if video has audio)
        if video.audio is not None:
            print("Video has existing audio. Mixing with background music.")
            # Mix the original audio with background music
            final_audio = CompositeAudioClip([video.audio, background_music])
        else:
            print("Video has no audio. Adding background music only.")
            # Use only the background music
            final_audio = background_music

        # Set the final audio to the video
        final_video = video.with_audio(final_audio)

        # Create output path with custom name
        output_dir = "final-output"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = f"{custom_video_name}.mp4"
        output_path = os.path.join(output_dir, output_filename)

        print(f"Writing final video with music: {output_path}")
        final_video.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True,
            logger=None,  # Suppress moviepy logs
        )

        # Clean up
        video.close()
        background_music.close()
        final_video.close()

        # Get file info
        file_size = os.path.getsize(output_path)
        print("Final video with music created successfully!")
        print(f"  Path: {output_path}")
        print(f"  Size: {file_size:,} bytes")

        return {"final_video_with_music_path": output_path}

    except Exception as e:
        print(f"Error adding music to video: {e}")
        return state


# Build the graph
def build_video_generation_graph():
    """
    Build the complete video generation workflow graph
    """
    builder = StateGraph(OverallState)

    # Add nodes
    builder.add_node("collect_file_to_process", collect_file_to_process)
    builder.add_node("generate_prompt", generate_prompt)
    builder.add_node("check_prompt_consistency", check_prompt_consistency)
    builder.add_node("generate_video", generate_video)
    builder.add_node("combine_videos", combine_videos)
    builder.add_node("generate_music", generate_music)
    builder.add_node("add_music_to_video", add_music_to_video)

    # Add edges
    builder.add_edge(START, "collect_file_to_process")
    builder.add_conditional_edges(
        "collect_file_to_process", continue_to_prompt_generation, ["generate_prompt"]
    )
    builder.add_edge("generate_prompt", "check_prompt_consistency")
    builder.add_conditional_edges(
        "check_prompt_consistency", continue_to_video_generation, ["generate_video"]
    )
    builder.add_edge("generate_video", "combine_videos")
    builder.add_edge("combine_videos", "generate_music")
    builder.add_edge("generate_music", "add_music_to_video")
    builder.add_edge("add_music_to_video", END)

    return builder.compile()


# Create the graph
video_graph = build_video_generation_graph()


def main():
    video_graph.invoke({"files_to_process": []})


if __name__ == "__main__":
    main()
