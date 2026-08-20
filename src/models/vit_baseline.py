import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOViTBaseline(nn.Module):
    """
    DINO ViT-B/16 backbone used as the standard-attention baseline.

    Input:
        images: Tensor [B, 3, 224, 224]

    Output:
        embeddings: Tensor [B, 768]

    The model uses the original standard self-attention layers
    of DINO ViT-B/16.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze: bool = False,
    ):
        super().__init__()

        self.backbone = torch.hub.load(
            "facebookresearch/dino:main",
            "dino_vitb16",
            pretrained=pretrained,
        )

        self.embedding_dim = 768

        if freeze:
            self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract DINO CLS-token embeddings.
        """
        return self.backbone(x)

    def embed(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Extract image embeddings.

        If normalize=True, embeddings are L2-normalized so that
        cosine similarity becomes a simple dot product.
        """
        z = self.forward(x)

        if normalize:
            z = F.normalize(z, p=2, dim=-1)

        return z

    def freeze_backbone(self):
        """
        Disable gradient computation for the complete ViT backbone.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self):
        """
        Enable gradient computation for the complete ViT backbone.
        """
        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def get_transformer_blocks(self):
        """
        Return the sequence of Transformer blocks.

        Useful later when replacing standard attention modules with
        efficient-attention variants.
        """
        return self.backbone.blocks