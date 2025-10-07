# Placeholder for a general multimodal combo: we compose vision encoder + text encoder/decoder
# For our pipeline, we only need embeddings (Φ_M, Ψ_M), not full generation.
import torch

class MMCompose:
    def __init__(self, vision_enc, text_enc):
        self.vision_enc=vision_enc
        self.text_enc=text_enc

    @torch.no_grad()
    def phi_image(self, pixel_batch):
        return self.vision_enc.embed_image(pixel_batch)

    @torch.no_grad()
    def psi_text(self, texts):
        return self.text_enc.embed_text(texts)
