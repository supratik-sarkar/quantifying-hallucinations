import torch
from transformers import CLIPProcessor, CLIPModel

class CLIPWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cuda"):
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.proc  = CLIPProcessor.from_pretrained(model_name)
        self.device = device

    @torch.no_grad()
    def embed_image(self, pixel_batch):
        # pixel_batch: (B,3,H,W) float in [0,1]
        inputs = self.proc(images=[(p*255).byte().permute(1,2,0).cpu().numpy() for p in pixel_batch],
                           return_tensors="pt", padding=True).to(self.device)
        return self.model.get_image_features(**inputs)

    @torch.no_grad()
    def embed_text(self, texts):
        inputs = self.proc(text=texts, return_tensors="pt", padding=True).to(self.device)
        return self.model.get_text_features(**inputs)

