"""
model.py
========
Unified Adaptive Framework for Multimodal Data Fusion
with Static Modality Reweighting (Phase 1)

Architecture:
    CLIP Image Encoder ──┐
                         ├──► Cross-Modal Attention ──► Static Gating ──► Classifier ──► 0/1
    CLIP Text Encoder  ──┘

Phase 1: Static Gating  — one learned scalar alpha for all samples
Phase 2: Dynamic Gating — per-sample alpha (add after Phase 1 is working)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel


# ─────────────────────────────────────────────────────────────────
# BLOCK 1: Cross-Modal Attention
# ─────────────────────────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention.
    - Image attends to Text  : "what text context is relevant to what I see?"
    - Text  attends to Image : "what visual context is relevant to what I say?"

    Each direction has:
        MultiheadAttention -> Residual + LayerNorm -> FFN -> Residual + LayerNorm
    """
    def __init__(self, embed_dim=512, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()

        # Attention layers (bidirectional)
        self.img_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.text_to_img_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # LayerNorms after attention
        self.norm_img_1  = nn.LayerNorm(embed_dim)
        self.norm_text_1 = nn.LayerNorm(embed_dim)

        # Feed-Forward Networks (FFN)
        self.ffn_img = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.ffn_text = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout)
        )

        # LayerNorms after FFN
        self.norm_img_2  = nn.LayerNorm(embed_dim)
        self.norm_text_2 = nn.LayerNorm(embed_dim)

    def forward(self, text_feat, img_feat):
        """
        Args:
            text_feat : (B, 512)
            img_feat  : (B, 512)
        Returns:
            text_out  : (B, 512)
            img_out   : (B, 512)
        """
        # Add sequence dimension: (B, 512) -> (B, 1, 512)
        t = text_feat.unsqueeze(1)
        i = img_feat.unsqueeze(1)

        # Image (Q) attends to Text (K, V)
        img_attn,  _ = self.img_to_text_attn(query=i, key=t, value=t)
        # Text (Q) attends to Image (K, V)
        text_attn, _ = self.text_to_img_attn(query=t, key=i, value=i)

        # Residual + LayerNorm 1
        img_out  = self.norm_img_1 (i + img_attn ).squeeze(1)
        text_out = self.norm_text_1(t + text_attn).squeeze(1)

        # FFN + Residual + LayerNorm 2
        img_out  = self.norm_img_2 (img_out  + self.ffn_img (img_out ))
        text_out = self.norm_text_2(text_out + self.ffn_text(text_out))

        return text_out, img_out


# ─────────────────────────────────────────────────────────────────
# BLOCK 2: Static Gating Network (Phase 1)
# ─────────────────────────────────────────────────────────────────

class StaticGatingNetwork(nn.Module):
    """
    Phase 1 — Static Reweighting.

    Learns ONE fixed alpha scalar for the entire dataset.
    - Same alpha applied to every sample
    - alpha close to 1.0 → trust image more globally
    - alpha close to 0.0 → trust text more globally

    This gives a stable, interpretable baseline before
    introducing per-sample dynamic gating in Phase 2.
    """
    def __init__(self):
        super().__init__()
        # Single learnable scalar, initialized at 0.5 (equal weight)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, text_feat, img_feat):
        """
        Args:
            text_feat : (B, 512)
            img_feat  : (B, 512)
        Returns:
            fused     : (B, 512)
            alpha     : scalar tensor — same for all samples
        """
        # Clamp to valid range (0, 1)
        alpha = torch.clamp(self.alpha, 0.0, 1.0)

        # Fixed weighted fusion — same alpha for every sample in the batch
        fused = alpha * img_feat + (1 - alpha) * text_feat   # (B, 512)

        return fused, alpha


# ─────────────────────────────────────────────────────────────────
# BLOCK 3: Full Model
# ─────────────────────────────────────────────────────────────────

class AdaptiveFusionModel(nn.Module):
    """
    Full pipeline:
    1. CLIP encodes image -> img_feat  (512)
    2. CLIP encodes text  -> text_feat (512)
    3. L2 normalize both
    4. CrossModalAttention (bidirectional + FFN)
    5. Re-normalize after attention
    6. StaticGatingNetwork — one learned alpha for all samples
    7. Classifier on [fused || text_feat || img_feat] -> logit
    """
    def __init__(
        self,
        embed_dim   = 512,
        num_heads   = 8,
        ffn_dim     = 1024,
        dropout     = 0.1,
        freeze_clip = True
    ):
        super().__init__()

        # CLIP backbone
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False

        # Cross-Modal Attention
        self.cross_attn = CrossModalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            dropout=dropout
        )

        # Static Gating (Phase 1)
        self.gating = StaticGatingNetwork()

        # Classifier: fused(512) + text(512) + img(512) = 1536
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, input_ids, attention_mask, pixel_values):
        """
        Args:
            input_ids      : (B, 77)
            attention_mask : (B, 77)
            pixel_values   : (B, 3, 224, 224)
        Returns:
            logit : (B,)
            alpha : scalar tensor (same for all samples in Phase 1)
        """
        # ── Step 1: Encode ────────────────────────────────────────
        text_out  = self.clip.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        text_feat = self.clip.text_projection(text_out.pooler_output)   # (B, 512)

        img_out   = self.clip.vision_model(pixel_values=pixel_values)
        img_feat  = self.clip.visual_projection(img_out.pooler_output)  # (B, 512)

        # ── Step 2: L2 Normalize ──────────────────────────────────
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat  = F.normalize(img_feat,  dim=-1)

        # ── Step 3: Cross-Modal Attention ─────────────────────────
        text_feat, img_feat = self.cross_attn(text_feat, img_feat)

        # ── Step 4: Re-normalize after attention ──────────────────
        text_feat = F.normalize(text_feat, dim=-1)
        img_feat  = F.normalize(img_feat,  dim=-1)

        # ── Step 5: Static Gating ─────────────────────────────────
        fused, alpha = self.gating(text_feat, img_feat)

        # ── Step 6: Classify ──────────────────────────────────────
        combined = torch.cat([fused, text_feat, img_feat], dim=-1)  # (B, 1536)
        logit    = self.classifier(combined).squeeze(1)              # (B,)

        return logit, alpha
