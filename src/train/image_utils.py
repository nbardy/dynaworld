import os
from io import BytesIO

import requests
from PIL import Image


def fetch_image(path_or_url):
    if path_or_url and os.path.exists(path_or_url):
        return Image.open(path_or_url).convert("RGB")
    if path_or_url and path_or_url.startswith("http"):
        return Image.open(BytesIO(requests.get(path_or_url).content)).convert("RGB")
    print("No valid local image provided, downloading a test puppy image...")
    url = "https://images.unsplash.com/photo-1543852786-1cf6624b9987?auto=format&fit=crop&w=400&q=80"
    return Image.open(BytesIO(requests.get(url).content)).convert("RGB")
