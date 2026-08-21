from src.models.vit_baseline import DINOViTBaseline

from src.models.attention.spatial_reduction import (
    SpatialReductionAttention,
)


class DINOViTEfficient(DINOViTBaseline):
    """
    DINO ViT-B/16 with selected standard attention modules replaced
    by PVT-inspired Spatial Reduction Attention.
    """

    def __init__(
        self,
        pretrained=True,
        freeze=False,
        sr_ratio=2,
        efficient_layers=None,
    ):

        # ---------------------------------------------------------
        # Load the original pretrained DINO ViT.
        # ---------------------------------------------------------

        super().__init__(
            pretrained=pretrained,
            freeze=False,
        )

        self.sr_ratio = sr_ratio

        num_blocks = len(
            self.backbone.blocks
        )

        # ---------------------------------------------------------
        # Default:
        # replace attention in ALL transformer blocks.
        # ---------------------------------------------------------

        if efficient_layers is None:

            efficient_layers = list(
                range(num_blocks)
            )

        self.efficient_layers = list(
            efficient_layers
        )

        # ---------------------------------------------------------
        # Validate indices
        # ---------------------------------------------------------

        for layer_index in (
            self.efficient_layers
        ):

            if not (
                0
                <= layer_index
                < num_blocks
            ):

                raise ValueError(
                    f"Invalid transformer layer "
                    f"{layer_index}. "
                    f"Valid range is "
                    f"0-{num_blocks - 1}."
                )

        # ---------------------------------------------------------
        # Replace attention modules.
        # ---------------------------------------------------------

        self._replace_attention()

        if freeze:
            self.freeze_backbone()

    # =================================================================
    # ATTENTION REPLACEMENT
    # =================================================================

    def _replace_attention(self):

        for layer_index in (
            self.efficient_layers
        ):

            block = (
                self.backbone
                .blocks[layer_index]
            )

            original_attention = (
                block.attn
            )

            efficient_attention = (
                SpatialReductionAttention
                .from_dino_attention(
                    original_attention,
                    sr_ratio=self.sr_ratio,
                )
            )

            block.attn = (
                efficient_attention
            )