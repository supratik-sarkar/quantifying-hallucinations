import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class TextBackbone:
    """
    Text encoder that returns a single embedding per input by mean-pooling
    the last hidden states from the encoder stack.

    Works with T5-style seq2seq models:
      - use self.model.get_encoder()(input_ids=..., attention_mask=...)
    """
    def __init__(self, name: str, device="cuda", max_length: int = 128):
        self.tok = AutoTokenizer.from_pretrained(name)
        # Ensure a pad token exists for batching
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token if self.tok.eos_token else self.tok.unk_token
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name).to(device)
        self.model.eval()
        self.device = device
        self.max_length = max_length

    @torch.no_grad()
    def embed_text(self, texts):
        """
        Returns a tensor of shape (B, D). Uses the encoder stack only.
        """
        enc_inputs = self.tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)

        # For T5-like models, use the encoder directly
        encoder = self.model.get_encoder()
        outputs = encoder(
            input_ids=enc_inputs["input_ids"],
            attention_mask=enc_inputs.get("attention_mask", None)
        )
        hidden = outputs.last_hidden_state  # (B, L, D)
        # Mean-pool over sequence length, masking pads if attention_mask is available
        attn = enc_inputs.get("attention_mask", None)
        if attn is not None:
            attn = attn.unsqueeze(-1).type_as(hidden)  # (B, L, 1)
            summed = (hidden * attn).sum(dim=1)
            denom = attn.sum(dim=1).clamp_min(1.0)
            emb = summed / denom
        else:
            emb = hidden.mean(dim=1)
        return emb
