from src.models.vit_baseline import DINOViTBaseline

from src.models.attention.metaformer_pooling import (
    MetaFormerPoolingTokenMixer,
)


class DINOViTMetaFormer(DINOViTBaseline):
    """
    DINO ViT-B/16 with selected self-attention modules replaced
    by a PoolFormer-style MetaFormer token mixer.

    By default all 12 Transformer blocks are modified.
    """

    def __init__(
        self,
        pretrained=True,
        freeze=False,
        pool_size=3,
        metaformer_layers=None,
    ):

        # ---------------------------------------------------------
        # Load pretrained DINO
        # ---------------------------------------------------------

        super().__init__(
            pretrained=pretrained,
            freeze=False,
        )

        self.pool_size = pool_size

        num_blocks = len(
            self.backbone.blocks
        )

        # ---------------------------------------------------------
        # Default -> replace ALL attention layers
        # ---------------------------------------------------------

        if metaformer_layers is None:

            metaformer_layers = list(
                range(num_blocks)
            )

        self.metaformer_layers = list(
            metaformer_layers
        )

        # ---------------------------------------------------------
        # Validate layer indices
        # ---------------------------------------------------------

        for layer_index in (
            self.metaformer_layers
        ):

            if not (
                0
                <= layer_index
                < num_blocks
            ):

                raise ValueError(
                    f"Invalid Transformer layer "
                    f"{layer_index}. "
                    f"Valid range is "
                    f"0-{num_blocks - 1}."
                )

        # ---------------------------------------------------------
        # Replace attention
        # ---------------------------------------------------------

        self._replace_attention()

        if freeze:
            self.freeze_backbone()

    def _replace_attention(self):

        for layer_index in (
            self.metaformer_layers
        ):

            block = (
                self.backbone
                .blocks[layer_index]
            )

            block.attn = (
                MetaFormerPoolingTokenMixer(
                    pool_size=self.pool_size
                )
            )