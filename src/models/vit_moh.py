from src.models.vit_baseline import (
    DINOViTBaseline,
)

from src.models.attention.mixture_of_heads import (
    MixtureOfHeadsAttention,
)


class DINOViTMoH(DINOViTBaseline):
    """
    DINO ViT-B/16 with selected standard attention layers replaced
    by Mixture-of-Head Attention.

    Example:

        moh_layers=None
            -> MoH Full

        moh_layers=list(range(6, 12))
            -> MoH Half
    """

    def __init__(
        self,
        pretrained=True,
        freeze=False,
        shared_heads=2,
        routed_heads=4,
        moh_layers=None,
    ):

        # =============================================================
        # LOAD ORIGINAL PRETRAINED DINO
        # =============================================================

        super().__init__(
            pretrained=pretrained,
            freeze=False,
        )

        self.shared_heads = (
            shared_heads
        )

        self.routed_heads = (
            routed_heads
        )

        num_blocks = len(
            self.backbone.blocks
        )

        # =============================================================
        # DEFAULT -> FULL
        # =============================================================

        if moh_layers is None:

            moh_layers = list(
                range(num_blocks)
            )

        self.moh_layers = list(
            moh_layers
        )

        # =============================================================
        # VALIDATE
        # =============================================================

        for layer_index in (
            self.moh_layers
        ):

            if not (
                0
                <= layer_index
                < num_blocks
            ):

                raise ValueError(
                    f"Invalid MoH layer "
                    f"{layer_index}. "
                    f"Valid range is "
                    f"0-{num_blocks - 1}."
                )

        # =============================================================
        # REPLACE ATTENTION
        # =============================================================

        self._replace_attention()

        if freeze:

            self.freeze_backbone()


    def _replace_attention(
        self,
    ):

        for layer_index in (
            self.moh_layers
        ):

            block = (
                self.backbone
                .blocks[
                    layer_index
                ]
            )

            original_attention = (
                block.attn
            )

            block.attn = (
                MixtureOfHeadsAttention
                .from_dino_attention(
                    original_attention,
                    shared_heads=(
                        self.shared_heads
                    ),
                    routed_heads=(
                        self.routed_heads
                    ),
                )
            )


    # =================================================================
    # AUXILIARY LOAD-BALANCING LOSS
    # =================================================================

    def get_moh_aux_loss(
        self,
    ):

        losses = []

        for layer_index in (
            self.moh_layers
        ):

            attention = (
                self.backbone
                .blocks[
                    layer_index
                ]
                .attn
            )

            if (
                attention.last_aux_loss
                is not None
            ):

                losses.append(
                    attention
                    .last_aux_loss
                )

        if not losses:

            return None

        return sum(losses) / len(
            losses
        )