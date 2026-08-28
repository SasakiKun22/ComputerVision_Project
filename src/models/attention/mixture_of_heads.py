import torch
import torch.nn as nn
import torch.nn.functional as F


class MixtureOfHeadsAttention(nn.Module):
    """
    MoH-inspired attention adapted to DINO ViT.

    The attention heads are divided into:

        shared heads:
            always active

        routed heads:
            dynamically selected for each token using top-k routing

    Example for DINO ViT-B/16:

        num_heads       = 12
        shared_heads    = 2
        routed_heads    = 4

    Therefore each token uses conceptually:

        2 shared + 4 routed = 6 / 12 heads

    The QKV and output projection weights can be initialized directly
    from the original pretrained DINO attention layer.

    NOTE:
    This implementation uses dense PyTorch operations for QKV and
    attention computation. Routing sparsifies the contribution of
    the attention heads, but does not use a custom sparse CUDA kernel.
    """

    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        shared_heads=2,
        routed_heads=4,
    ):

        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim={dim} must be divisible by "
                f"num_heads={num_heads}"
            )

        if shared_heads < 0:
            raise ValueError(
                "shared_heads must be >= 0"
            )

        if routed_heads < 0:
            raise ValueError(
                "routed_heads must be >= 0"
            )

        if shared_heads > num_heads:
            raise ValueError(
                "shared_heads cannot exceed num_heads"
            )

        num_routed_candidates = (
            num_heads - shared_heads
        )

        if routed_heads > num_routed_candidates:
            raise ValueError(
                "routed_heads cannot exceed the number "
                "of non-shared heads"
            )

        if shared_heads + routed_heads == 0:
            raise ValueError(
                "At least one attention head must be active"
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = (
            dim // num_heads
        )

        self.scale = (
            qk_scale
            if qk_scale is not None
            else self.head_dim ** -0.5
        )

        self.shared_heads = (
            shared_heads
        )

        self.routed_heads = (
            routed_heads
        )

        self.num_routed_candidates = (
            num_routed_candidates
        )

        # =============================================================
        # STANDARD DINO ATTENTION PARAMETERS
        # =============================================================

        self.qkv = nn.Linear(
            dim,
            dim * 3,
            bias=qkv_bias,
        )

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

        # =============================================================
        # ROUTERS
        # =============================================================

        # -------------------------------------------------------------
        # Router for candidate routed heads
        # -------------------------------------------------------------

        if self.routed_heads > 0:

            self.routed_router = nn.Linear(
                dim,
                self.num_routed_candidates,
                bias=False,
            )

            nn.init.normal_(
                self.routed_router.weight,
                mean=0.0,
                std=0.02,
            )

        else:

            self.routed_router = None

        # -------------------------------------------------------------
        # Router for shared heads.
        #
        # With >1 shared head, allows different shared heads to receive
        # different weights.
        # -------------------------------------------------------------

        if self.shared_heads > 1:

            self.shared_router = nn.Linear(
                dim,
                self.shared_heads,
                bias=False,
            )

            nn.init.normal_(
                self.shared_router.weight,
                mean=0.0,
                std=0.02,
            )

        else:

            self.shared_router = None

        # -------------------------------------------------------------
        # Router between shared group and routed group
        # -------------------------------------------------------------

        if (
            self.shared_heads > 0
            and self.routed_heads > 0
        ):

            self.group_router = nn.Linear(
                dim,
                2,
                bias=False,
            )

            nn.init.normal_(
                self.group_router.weight,
                mean=0.0,
                std=0.02,
            )

        else:

            self.group_router = None

        # Last load-balancing auxiliary loss.
        #
        # It is updated during each forward.
        self.last_aux_loss = None


    # =================================================================
    # ROUTING
    # =================================================================

    def _compute_head_gates(
        self,
        x,
    ):
        """
        Returns:

            head_gates:
                [B, N, H]

        where inactive routed heads have gate = 0.
        """

        B, N, C = x.shape

        # =============================================================
        # SHARED HEADS
        # =============================================================

        shared_gates = None

        if self.shared_heads > 0:

            if self.shared_heads == 1:

                shared_gates = torch.ones(
                    B,
                    N,
                    1,
                    device=x.device,
                    dtype=x.dtype,
                )

            else:

                shared_logits = (
                    self.shared_router(x)
                )

                shared_gates = F.softmax(
                    shared_logits,
                    dim=-1,
                )

                # Preserve approximately unit scale
                # for each active head.
                shared_gates = (
                    shared_gates
                    * self.shared_heads
                )

        # =============================================================
        # ROUTED HEADS
        # =============================================================

        routed_gates = None

        if self.routed_heads > 0:

            routed_logits = (
                self.routed_router(x)
            )

            routing_probabilities = (
                F.softmax(
                    routed_logits,
                    dim=-1,
                )
            )

            # ---------------------------------------------------------
            # Top-k routed heads for every token
            # ---------------------------------------------------------

            _, topk_indices = torch.topk(
                routing_probabilities,
                k=self.routed_heads,
                dim=-1,
            )

            routing_mask = torch.zeros_like(
                routing_probabilities
            )

            routing_mask.scatter_(
                dim=-1,
                index=topk_indices,
                value=1.0,
            )

            routed_gates = (
                routing_probabilities
                * routing_mask
            )

            # ---------------------------------------------------------
            # Re-normalize probabilities over active routed heads
            # ---------------------------------------------------------

            denominator = (
                routed_gates.sum(
                    dim=-1,
                    keepdim=True,
                )
                .clamp_min(1e-8)
            )

            routed_gates = (
                routed_gates
                / denominator
            )

            # Average active gate approximately 1.
            routed_gates = (
                routed_gates
                * self.routed_heads
            )

            # ---------------------------------------------------------
            # MoE-style load balancing loss
            # ---------------------------------------------------------

            if self.training:

                importance = (
                    routing_probabilities
                    .mean(
                        dim=(0, 1)
                    )
                )

                load = (
                    routing_mask
                    .float()
                    .mean(
                        dim=(0, 1)
                    )
                )

                num_experts = (
                    self.num_routed_candidates
                )

                self.last_aux_loss = (
                    num_experts
                    * torch.sum(
                        importance
                        * load
                    )
                )

            else:

                self.last_aux_loss = None

        # =============================================================
        # SHARED / ROUTED GROUP WEIGHTING
        # =============================================================

        if (
            shared_gates is not None
            and routed_gates is not None
        ):

            group_logits = (
                self.group_router(x)
            )

            group_gates = F.softmax(
                group_logits,
                dim=-1,
            )

            # Same scale convention used by MoH:
            # average group coefficient ~1.
            group_gates = (
                group_gates * 2.0
            )

            shared_gates = (
                shared_gates
                * group_gates[
                    ...,
                    0:1
                ]
            )

            routed_gates = (
                routed_gates
                * group_gates[
                    ...,
                    1:2
                ]
            )

            head_gates = torch.cat(
                [
                    shared_gates,
                    routed_gates,
                ],
                dim=-1,
            )

        elif shared_gates is not None:

            head_gates = (
                shared_gates
            )

        else:

            head_gates = (
                routed_gates
            )

        return head_gates


    # =================================================================
    # FORWARD
    # =================================================================

    def forward(
        self,
        x,
    ):

        B, N, C = x.shape

        # =============================================================
        # ROUTING
        # =============================================================

        head_gates = (
            self._compute_head_gates(
                x
            )
        )

        # =============================================================
        # Q K V
        # =============================================================

        qkv = (
            self.qkv(x)
            .reshape(
                B,
                N,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(
                2,
                0,
                3,
                1,
                4,
            )
        )

        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        # q, k, v:
        #
        # [B, H, N, D]

        # =============================================================
        # STANDARD SCALED DOT-PRODUCT ATTENTION
        # =============================================================

        attention = (
            q
            @ k.transpose(
                -2,
                -1,
            )
        )

        attention = (
            attention
            * self.scale
        )

        attention = (
            attention.softmax(
                dim=-1
            )
        )

        attention = (
            self.attn_drop(
                attention
            )
        )

        # =============================================================
        # HEAD OUTPUTS
        # =============================================================

        head_output = (
            attention @ v
        )

        # [B, H, N, D]
        # ->
        # [B, N, H, D]

        head_output = (
            head_output
            .transpose(
                1,
                2,
            )
        )

        # =============================================================
        # MIXTURE-OF-HEAD ROUTING
        # =============================================================

        head_output = (
            head_output
            * head_gates.unsqueeze(
                -1
            )
        )

        # =============================================================
        # CONCATENATE HEADS
        # =============================================================

        output = (
            head_output
            .reshape(
                B,
                N,
                C,
            )
        )

        output = (
            self.proj(
                output
            )
        )

        output = (
            self.proj_drop(
                output
            )
        )

        # =============================================================
        # EFFECTIVE ATTENTION MAP
        #
        # Keep DINO-compatible tuple interface.
        #
        # Gate each query/head attention map so inactive routed heads
        # appear as zero.
        # =============================================================

        effective_attention = (
            attention
            * head_gates
            .permute(
                0,
                2,
                1,
            )
            .unsqueeze(-1)
        )

        return (
            output,
            effective_attention,
        )


    # =================================================================
    # INITIALIZE FROM PRETRAINED DINO
    # =================================================================

    @classmethod
    def from_dino_attention(
        cls,
        original_attention,
        shared_heads=2,
        routed_heads=4,
    ):

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

        module = cls(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=(
                original_attention.scale
            ),
            attn_drop=(
                original_attention
                .attn_drop
                .p
            ),
            proj_drop=(
                original_attention
                .proj_drop
                .p
            ),
            shared_heads=shared_heads,
            routed_heads=routed_heads,
        )

        # -------------------------------------------------------------
        # Preserve DINO pretrained QKV weights
        # -------------------------------------------------------------

        module.qkv.weight.data.copy_(
            original_attention
            .qkv
            .weight
            .data
        )

        if (
            original_attention.qkv.bias
            is not None
        ):

            module.qkv.bias.data.copy_(
                original_attention
                .qkv
                .bias
                .data
            )

        # -------------------------------------------------------------
        # Preserve pretrained output projection
        # -------------------------------------------------------------

        module.proj.weight.data.copy_(
            original_attention
            .proj
            .weight
            .data
        )

        module.proj.bias.data.copy_(
            original_attention
            .proj
            .bias
            .data
        )

        return module