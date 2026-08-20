import torch
import torch.nn as nn


class PerceptualSimilarityModel(nn.Module):
    """
    DreamSim-like perceptual similarity pipeline.

    The model receives an image encoder and compares images
    using cosine distance between L2-normalized embeddings.

    Target convention used by NIGHTS:
        0 -> LEFT preferred
        1 -> RIGHT preferred
    """

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        """
        Convert images into L2-normalized embeddings.

        Input:
            images: [B, 3, H, W]

        Output:
            embeddings: [B, D]
        """
        return self.encoder.embed(
            images,
            normalize=True,
        )

    @staticmethod
    def cosine_distance(
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cosine distance between L2-normalized embeddings.

        Since ||z1|| = ||z2|| = 1:

            cosine_similarity = z1 · z2

            cosine_distance = 1 - similarity
        """

        similarity = torch.sum(
            z1 * z2,
            dim=-1,
        )

        return 1.0 - similarity

    def compare_embeddings(
        self,
        reference_embedding: torch.Tensor,
        left_embedding: torch.Tensor,
        right_embedding: torch.Tensor,
    ):
        """
        Compare already-computed embeddings.
        """

        d_left = self.cosine_distance(
            reference_embedding,
            left_embedding,
        )

        d_right = self.cosine_distance(
            reference_embedding,
            right_embedding,
        )

        # NIGHTS convention:
        # 0 -> LEFT preferred
        # 1 -> RIGHT preferred
        prediction = (
            d_right < d_left
        ).long()

        return {
            "distance_left": d_left,
            "distance_right": d_right,
            "prediction": prediction,
        }

    def forward(
        self,
        reference: torch.Tensor,
        left: torch.Tensor,
        right: torch.Tensor,
    ):
        """
        Compare a batch of NIGHTS triplets.

        To avoid three independent ViT forward passes,
        all images are concatenated and encoded together.
        """

        batch_size = reference.shape[0]

        images = torch.cat(
            [reference, left, right],
            dim=0,
        )

        embeddings = self.encode(images)

        z_reference = embeddings[:batch_size]

        z_left = embeddings[
            batch_size:2 * batch_size
        ]

        z_right = embeddings[
            2 * batch_size:
        ]

        return self.compare_embeddings(
            z_reference,
            z_left,
            z_right,
        )