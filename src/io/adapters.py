import torch, torchvision.transforms as T
from PIL import Image
from io import BytesIO
import requests

IMG_SIZE=224

def fetch_image_maybe(url):
    if url is None: return None
    try:
        img = Image.open(requests.get(url, timeout=3).content if isinstance(url,str) else url).convert("RGB")
        return img
    except Exception:
        return None

def default_image_transform():
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor()
    ])

def collate_vision_text(batch, text_key="text", image_key="image_url"):
    imgs=[]; texts=[]
    tfm = default_image_transform()
    for b in batch:
        im = fetch_image_maybe(b.get(image_key))
        if im is None:
            # generate a simple synthetic image (colored square)
            im = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(int(b["id"])%255,50,100))
        imgs.append(tfm(im))
        texts.append(b.get(text_key, ""))
    return torch.stack(imgs,0), texts

def collate_audio_text(batch, caption_key="caption"):
    # audio handled as text-like embedding via caption or synthetic token string
    captions = [b.get(caption_key, f"audio {b['id']}") for b in batch]
    return captions
