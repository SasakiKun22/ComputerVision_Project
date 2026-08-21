import math

import torch
import torch.nn as nn


class SpatialReductionAttention(nn.Module):
    """
    PVT-inspired pooled Spatial Reduction Attention for DINO ViT.

    Queries are computed from the full token sequence.

    Keys and values are computed from a spatially reduced sequence:
        14x14 patch tokens -> 7x7 patch tokens for sr_ratio=2

    The CLS token is preserved.

    This module has the same forward interface as the original
    DINO Attention module:

        output, attention_map = attention(x)
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        sr_ratio=2,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        if sr_ratio < 1:
            raise ValueError(
                "sr_ratio must be >= 1"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.sr_ratio = sr_ratio

        head_dim = dim // num_heads

        self.scale = (
            qk_scale
            if qk_scale is not None
            else head_dim ** -0.5
        )

        # ---------------------------------------------------------
        # Separate Q and KV projections
        #
        # This lets us compute Q on all tokens but K,V only on the
        # spatially reduced token sequence.
        # ---------------------------------------------------------

        self.q = nn.Linear(
            dim,
            dim,
            bias=qkv_bias,
        )

        self.kv = nn.Linear(
            dim,
            dim * 2,
            bias=qkv_bias,
        )

        # ---------------------------------------------------------
        # Spatial reduction
        # ---------------------------------------------------------

        if sr_ratio > 1:

            self.pool = nn.AvgPool2d(
                kernel_size=sr_ratio,
                stride=sr_ratio,
            )

            self.norm = nn.LayerNorm(
                dim
            )

        else:

            self.pool = None
            self.norm = None

        # ---------------------------------------------------------
        # Attention/output
        # ---------------------------------------------------------

        self.attn_drop = nn.Dropout(
            attn_drop
        )

        self.proj = nn.Linear(
            dim,
            dim,
        )

        self.proj_drop = nn.Dropout(
            proj_drop
        )

    # =================================================================
    # TOKEN REDUCTION
    # =================================================================

    def _reduce_tokens(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if self.sr_ratio == 1:
            return x

        B, N, C = x.shape

        # ---------------------------------------------------------
        # Separate CLS token from patch tokens.
        # ---------------------------------------------------------

        cls_token = x[:, :1, :]

        patch_tokens = x[:, 1:, :]

        num_patches = (
            patch_tokens.shape[1]
        )

        spatial_size = int(
            math.sqrt(num_patches)
        )

        if (
            spatial_size
            * spatial_size
            != num_patches
        ):
            raise ValueError(
                "SpatialReductionAttention expects "
                "a square patch-token grid. "
                f"Received {num_patches} patch tokens."
            )

        if (
            spatial_size
            % self.sr_ratio
            != 0
        ):
            raise ValueError(
                f"Patch grid {spatial_size}x{spatial_size} "
                f"is not divisible by "
                f"sr_ratio={self.sr_ratio}"
            )

        # ---------------------------------------------------------
        # [B, N, C]
        #      ->
        # [B, C, H, W]
        # ---------------------------------------------------------

        patch_tokens = (
            patch_tokens
            .transpose(1, 2)
            .reshape(
                B,
                C,
                spatial_size,
                spatial_size,
            )
        )

        # ---------------------------------------------------------
        # Spatial reduction
        #
        # sr_ratio=2:
        #
        # 14x14 -> 7x7
        # ---------------------------------------------------------

        patch_tokens = self.pool(
            patch_tokens
        )

        # ---------------------------------------------------------
        # Back to token representation:
        #
        # [B,C,H',W']
        #       ->
        # [B,N',C]
        # ---------------------------------------------------------

        patch_tokens = (
            patch_tokens
            .flatten(2)
            .transpose(1, 2)
        )

        patch_tokens = self.norm(
            patch_tokens
        )

        # Preserve the original CLS token.
        reduced_tokens = torch.cat(
            [
                cls_token,
                patch_tokens,
            ],
            dim=1,
        )

        return reduced_tokens

    # =================================================================
    # FORWARD
    # =================================================================

    def forward(
        self,
        x: torch.Tensor,
    ):

        B, N, C = x.shape

        # ---------------------------------------------------------
        # Queries
        #
        # Q uses ALL tokens.
        # ---------------------------------------------------------

        q = self.q(x)

        q = (
            q
            .reshape(
                B,
                N,
                self.num_heads,
                C // self.num_heads,
            )
            .permute(
                0,
                2,
                1,
                3,
            )
        )

        # ---------------------------------------------------------
        # Reduce tokens used for K and V.
        # ---------------------------------------------------------

        reduced_x = (
            self._reduce_tokens(x)
        )

        N_reduced = (
            reduced_x.shape[1]
        )

        # ---------------------------------------------------------
        # Keys and values
        # ---------------------------------------------------------

        kv = self.kv(
            reduced_x
        )

        kv = (
            kv
            .reshape(
                B,
                N_reduced,
                2,
                self.num_heads,
                C // self.num_heads,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        k = kv[0]
        v = kv[1]

        # ---------------------------------------------------------
        # Attention
        #
        # [B, heads, N, N_reduced]
        # ---------------------------------------------------------

        attention = (
            q
            @ k.transpose(-2, -1)
        )

        attention = (
            attention
            * self.scale
        )

        attention = attention.softmax(
            dim=-1
        )

        attention = self.attn_drop(
            attention
        )

        # ---------------------------------------------------------
        # Weighted values
        # ---------------------------------------------------------

        output = (
            attention
            @ v
        )

        output = (
            output
            .transpose(1, 2)
            .reshape(
                B,
                N,
                C,
            )
        )

        output = self.proj(
            output
        )

        output = self.proj_drop(
            output
        )

        # Same interface used by DINO Attention.
        return output, attention

    # =================================================================
    # INITIALIZATION FROM STANDARD DINO ATTENTION
    # =================================================================

    @classmethod
    def from_dino_attention(
        cls,
        original_attention,
        sr_ratio=2,
    ):
        """
        Create SpatialReductionAttention from an existing DINO
        standard Attention module.

        Q, K, V and output projection weights are copied from the
        pretrained DINO attention.

        Only the spatial-reduction operation is new.
        """

        dim = (
            original_attention
            .qkv
            .in_features
        )

        num_heads = (
            original_attention
            .num_heads
        )

        qkv_bias = (
            original_attention
            .qkv
            .bias
            is not None
        )

        new_attention = cls(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=original_attention.scale,
            attn_drop=original_attention.attn_drop.p,
            proj_drop=original_attention.proj_drop.p,
            sr_ratio=sr_ratio,
        )

        # ---------------------------------------------------------
        # Original DINO:
        #
        # qkv.weight:
        #
        # [ Q ]
        # [ K ]
        # [ V ]
        #
        # shape = [3*dim, dim]
        # ---------------------------------------------------------

        with torch.no_grad():

            qkv_weight = (
                original_attention
                .qkv
                .weight
            )

            # Q weights
            new_attention.q.weight.copy_(
                qkv_weight[:dim]
            )

            # K + V weights
            new_attention.kv.weight.copy_(
                qkv_weight[dim:]
            )

            # -----------------------------------------------------
            # Biases
            # -----------------------------------------------------

            if qkv_bias:

                qkv_bias_value = (
                    original_attention
                    .qkv
                    .bias
                )

                new_attention.q.bias.copy_(
                    qkv_bias_value[:dim]
                )

                new_attention.kv.bias.copy_(
                    qkv_bias_value[dim:]
                )

            # -----------------------------------------------------
            # Output projection
            # -----------------------------------------------------

            new_attention.proj.weight.copy_(
                original_attention
                .proj
                .weight
            )

            if (
                original_attention
                .proj
                .bias
                is not None
            ):

                new_attention.proj.bias.copy_(
                    original_attention
                    .proj
                    .bias
                )

        return new_attention