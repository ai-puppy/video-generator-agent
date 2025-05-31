# Video Generator Agent

A simple tool to generate videos from images.

## Setup

1. **Install uv**

   Download and install uv package manager from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)

2. **Install dependencies**

   ```bash
   uv sync
   ```

3. **Set up API keys**

   Refer to `example.env` to get the necessary API keys and copy them to `.env` file

## Usage

1. **Add your images**

   Put your pictures into the `pic-input` folder

2. **Generate video**

   ```bash
   python main.py
   ```

The generated video will be created and saved automatically.
