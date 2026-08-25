import math

import torch
import torch.nn as nn


class MetaFormerPoolingTokenMixer(nn.Module):
    """
    PoolFormer-style token mixer adapted to the DINO ViT token format.

    Input:
        [B, N, C]

    where:
        token 0   -> CLS token
        tokens 1: -> spatial patch tokens

    Patch tokens are mixed using the PoolFormer operation:

        AvgPool(x) - x

    The CLS token receives a global aggregation of the patch tokens.

    The module returns (output, None) in order to preserve the interface
    expected by the original DINO Transformer Block.
    """

    def __init__(
        self,
        pool_size: int = 3,
    ):
        super().__init__()

        if pool_size <= 0:
            raise ValueError(
                "pool_size must be > 0"
            )

        if pool_size % 2 == 0:
            raise ValueError(
                "pool_size must be odd"
            )

        self.pool_size = pool_size

        self.pool = nn.AvgPool2d(
            kernel_size=pool_size,
            stride=1,
            padding=pool_size // 2,
            count_include_pad=False,
        )

    def forward(
        self,
        x: torch.Tensor,
    ):

        B, N, C = x.shape

        # ---------------------------------------------------------
        # Separate CLS token and spatial patch tokens
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
            spatial_size * spatial_size
            != num_patches
        ):
            raise ValueError(
                "MetaFormerPoolingTokenMixer expects "
                "a square patch-token grid. "
                f"Received {num_patches} patch tokens."
            )

        # ---------------------------------------------------------
        # Convert patch tokens:
        #
        # [B, N, C]
        # ->
        # [B, C, H, W]
        # ---------------------------------------------------------

        patch_map = (
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
        # PoolFormer token mixing:
        #
        # AvgPool(x) - x
        # ---------------------------------------------------------

        pooled = self.pool(
            patch_map
        )

        mixed_patch_map = (
            pooled
            - patch_map
        )

        # ---------------------------------------------------------
        # Back to token representation
        # ---------------------------------------------------------

        mixed_patch_tokens = (
            mixed_patch_map
            .flatten(2)
            .transpose(1, 2)
        )

        # ---------------------------------------------------------
        # CLS token adaptation
        #
        # PoolFormer does not originally have a CLS token.
        # We inject global patch information into it.
        #
        # We use the same "mixed - original" idea used by
        # PoolFormer, but with global pooling.
        # ---------------------------------------------------------

        global_patch_token = (
            patch_tokens
            .mean(
                dim=1,
                keepdim=True,
            )
        )

        mixed_cls_token = (
            global_patch_token
            - cls_token
        )

        # ---------------------------------------------------------
        # Reassemble token sequence
        # ---------------------------------------------------------

        output = torch.cat(
            [
                mixed_cls_token,
                mixed_patch_tokens,
            ],
            dim=1,
        )

        # DINO Attention normally returns:
        #
        # output, attention_map
        #
        # Pooling has no attention map.
        return output, None