import operator
import os
from typing import Annotated

from typing_extensions import TypedDict


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
