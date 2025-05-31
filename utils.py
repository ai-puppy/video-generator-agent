import os
import requests
from urllib.parse import urlparse

def download_video(video_url, filename=None, output_dir="video-output"):
    """
    Download video from URL to local file
    
    Args:
        video_url (str): URL of the video to download
        filename (str, optional): Name for the downloaded file. If None, extracts from URL
        output_dir (str, optional): Directory to save the video. Defaults to "video-output"
    
    Returns:
        str: Full path of downloaded file if successful, None if failed
    """
    try:
        print(f"Downloading video from: {video_url}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get filename from URL if not provided
        if not filename:
            parsed_url = urlparse(video_url)
            filename = os.path.basename(parsed_url.path)
            if not filename or not filename.endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm')):
                filename = "downloaded_video.mp4"
        
        # Construct full file path
        file_path = os.path.join(output_dir, filename)
        
        # Download the video
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        # Save to file
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"Video downloaded successfully to: {file_path}")
        return file_path
        
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None