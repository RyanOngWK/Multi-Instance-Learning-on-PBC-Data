import torch
import torch.nn as nn
import timm
import sys
from typing import Dict, List, Optional, Tuple

def _clean_state_dict_keys(state_dict: dict) -> dict:
    cleaned = {}
    for k, v in state_dict.items():
        nk = k
        for prefix in ("module.", "model.", "backbone.", "encoder."):
            if nk.startswith(prefix):
                nk = nk[len(prefix):]
        cleaned[nk] = v
    return cleaned

class DINOv3ViTLEncoder(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super().__init__()
        self.model = timm.create_model(
            "hf_hub:timm/vit_large_patch16_dinov3.lvd1689m",
            img_size=224,
            pretrained=True,   # ← pulls weights from timm hub automatically
            num_classes=0,
        )
        self.embedding_dim = self.model.num_features  # 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        if feats.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(feats.shape)}")
        return feats
class DINOv3ConvNextEncoder(nn.Module):
    def __init__(self, checkpoint_path: str = None):
        super().__init__()
        self.model = timm.create_model(
            "hf_hub:timm/convnext_large.dinov3_lvd1689m",
            pretrained=True,
            num_classes=0,
            # img_size removed — ConvNeXt is fully convolutional, no fixed input size needed
        )
        self.embedding_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        if feats.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(feats.shape)}")
        return feats 
class RuiPathViTL16Encoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.model = timm.create_model(
            "vit_large_patch16_224",
            img_size=224,
            patch_size=16,
            init_values=1e-5,
            num_classes=0,
            dynamic_img_size=False,
            # pretrained= True
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = _clean_state_dict_keys(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] Missing encoder keys: {len(missing)}", file=sys.stderr)
        if unexpected:
            print(f"[WARN] Unexpected encoder keys: {len(unexpected)}", file=sys.stderr)
        self.embedding_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        if feats.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(feats.shape)}")
        return feats
class NextViTEncoder(nn.Module):
    DEFAULT_MODEL = "nextvit_base"

    def __init__(self, checkpoint_path: str, model_name: Optional[str] = None):
        super().__init__()
        name = model_name or self.DEFAULT_MODEL
        self.model = timm.create_model(name, pretrained=False, num_classes=0)

        state = torch.load(checkpoint_path, map_location="cpu")
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        state = _clean_state_dict_keys(state)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:
            print(f"[WARN] NextViT missing keys: {len(missing)}", file=sys.stderr)
        if unexpected:
            print(f"[WARN] NextViT unexpected keys: {len(unexpected)}", file=sys.stderr)
        self.embedding_dim = self.model.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        if feats.ndim != 2:
            raise ValueError(f"Expected [N, D], got {tuple(feats.shape)}")
        return feats
    
def build_encoder(
    encoder_type: str,
    checkpoint_path: str,
    model_name: Optional[str] = None,
) -> nn.Module:
    encoders = {
        "ruipath":     RuiPathViTL16Encoder,
        "nextvit":     NextViTEncoder,
        "dinov3vit": DINOv3ViTLEncoder,
        "dinov3convnext": DINOv3ConvNextEncoder,
        # "fastvit":     FastViTEncoder,
        # "mambavision": MambaVisionEncoder,
    }
    if encoder_type not in encoders:
        raise ValueError(
            f"Unknown encoder type '{encoder_type}'. "
            f"Choose from: {list(encoders.keys())}"
        )
    cls = encoders[encoder_type]
    # RuiPath doesn't accept model_name — handle separately
    if encoder_type in ("ruipath","dinov3vit","dinov3convnext"):
        return cls(checkpoint_path=checkpoint_path)
    return cls(checkpoint_path=checkpoint_path, model_name=model_name)
class AttentionMIL(nn.Module):
    def __init__(
        self,
        patch_encoder: nn.Module,
        embedding_dim: int,
        num_classes: int,
        attn_dim: int = 256,
        dropout: float = 0.25,
    ):
        super().__init__()
        self.patch_encoder = patch_encoder
        self.attn_V = nn.Sequential(nn.Linear(embedding_dim, attn_dim), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(embedding_dim, attn_dim), nn.Sigmoid())
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        h_feat = self.patch_encoder(x)
        h_feat = h_feat.view(b, t, -1)
        A_V = self.attn_V(h_feat)
        A_U = self.attn_U(h_feat)
        A = self.attn_w(A_V * A_U).squeeze(-1)          # (B, T)
        if mask is not None:
            # Set padding positions to -inf so softmax gives them ~0 weight
            A = A.masked_fill(mask == 0, float("-inf"))
        A = torch.softmax(A, dim=1)
        M = torch.bmm(A.unsqueeze(1), h_feat).squeeze(1)
        logits = self.classifier(M)
        return logits, A

