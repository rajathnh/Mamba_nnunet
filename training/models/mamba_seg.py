"""
MambaSeg2D - A 2D Segmentation Model for Pancreas CT using Vision Mamba with SS2D (Cross-Scan).

Designed to be a drop-in model for the nnUNet trainer pipeline.
It accepts (B, C, H, W) inputs and returns a list of multi-resolution logits
when in training mode (for nnUNet's deep supervision loss), or a single logit
tensor during validation/inference.

Architecture:
    1. Patch Embedding     : CNN-based tokenizer, (B,C,H,W) -> (B,D,H/4,W/4)
    2. Encoder             : 4x VSS Stages with PatchMerge downsampling, saves skip feats
    3. Bottleneck VSS      : Deepest feature processing
    4. Decoder             : Progressive upsampling with skip connections & Conv refinement
    5. Deep Super. Heads   : 1x1 Conv outputs at 3 decoder resolutions for the nnUNet loss

No external 'mamba-ssm' package required — fully local PyTorch implementation.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ─────────────────────────────────────────────────────────────────────────────
# 1. SS2D: 2D Selective Scan Module (Cross-Scan Core)
#    Implements 4-direction scanning so spatial 2D context is preserved fully.
# ─────────────────────────────────────────────────────────────────────────────

class SS2D(nn.Module):
    """
    2D Selective State Space Model block.
    Processes the image in 4 scan directions simultaneously:
        1. Top-left  -> Bottom-right  (rows, forward)
        2. Bottom-right -> Top-left    (rows, backward)
        3. Top-right -> Bottom-left   (columns, forward transposed)
        4. Bottom-left -> Top-right   (columns, backward transposed)
    Results are merged by summing the unscanned outputs for full 2D context.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)

        # Input projection: splits into two branches (x and z for gating)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local context convolution (causal depthwise conv in the sequence dim)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv // 2,
            groups=self.d_inner, bias=True
        )

        self.act = nn.SiLU()

        # SSM parameters — separate for each of the 4 scan directions
        dt_rank = math.ceil(d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # A, B, C, D SSM matrices
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # learned in log space for stability
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection back to d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(self.d_inner)

    def ssm_scan(self, u: torch.Tensor) -> torch.Tensor:
        """
        Parallel prefix scan for the linear SSM recurrence h_t = dA_t*h_{t-1} + dB_t*u_t.

        Uses log-space cumulative sum (torch.cumsum) instead of a Python for loop.
        This runs O(log L) depth fully on the GPU, keeping GPU utilization near 100%
        regardless of sequence length L = H*W.

        Closed-form solution (log-space trick):
            log_decay(t) = cumsum_{s=0..t}[ log(dA_s) ]          (prefix log-product of A)
            h_t = exp(log_decay_t) * cumsum_{s=0..t}[ dB_s*u_s / exp(log_decay_s) ]
        """
        B, L, D = u.shape
        dt_rank = math.ceil(self.d_model / 16)

        x_dbl = self.x_proj(u)                         # (B, L, dt_rank+2*d_state)
        dt, B_mat, C_mat = torch.split(x_dbl, [dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.softplus(self.dt_proj(dt))               # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())              # (d_inner, d_state) — always negative

        # ── Discretization ───────────────────────────────────────────────────
        # log(dA) = dt * A  (using log(exp(x)) = x, avoids exp then log)
        log_dA  = torch.einsum('bld,dn->bldn', dt, A)          # (B, L, d_inner, d_state)
        dB_u    = torch.einsum('bld,bln->bldn', dt, B_mat) \
                * u.unsqueeze(-1)                              # (B, L, d_inner, d_state)

        # ── Parallel Log-Space Prefix Scan ───────────────────────────────────
        # Step 1: prefix sum of log(dA) — equivalent to log(product of dA_0..dA_t)
        log_decay = torch.cumsum(log_dA, dim=1)                # (B, L, d_inner, d_state)
        decay     = torch.exp(log_decay)                       # (B, L, d_inner, d_state)

        # Step 2: divide each input contribution by its accumulated decay factor
        x_normalized = dB_u / (decay + 1e-12)                 # (B, L, d_inner, d_state)

        # Step 3: parallel prefix sum of normalised contributions
        x_cumsum = torch.cumsum(x_normalized, dim=1)           # (B, L, d_inner, d_state)

        # Step 4: multiply back by decay to get h_t at each position
        h = decay * x_cumsum                                   # (B, L, d_inner, d_state)

        # ── y_t = C_t . h_t (sum over d_state) ──────────────────────────────
        y = torch.einsum('bldn,bln->bld', h, C_mat)            # (B, L, d_inner)
        y = y + u * self.D                                     # skip connection (D term)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, d_model)  — spatial-last layout used by VSS block
        returns: (B, H, W, d_model)
        """
        B, H, W, C = x.shape

        # Split into value branch and gate branch
        xz = self.in_proj(x)                           # (B,H,W, 2*d_inner)
        x_val, z = xz.chunk(2, dim=-1)                 # each (B,H,W,d_inner)

        # Depthwise conv for local context (operates on the 2D spatial grid)
        x_val = rearrange(x_val, 'b h w d -> b d h w')
        x_val = self.act(self.conv2d(x_val))
        x_val = rearrange(x_val, 'b d h w -> b h w d')

        # ── Cross-Scan: flatten in 4 directions ──────────────────────────────
        # Direction 1: row-major forward  (top-left  -> bottom-right)
        s1 = rearrange(x_val, 'b h w d -> b (h w) d')
        # Direction 2: row-major backward (bottom-right -> top-left)
        s2 = s1.flip(1)
        # Direction 3: col-major forward  (top-right -> bottom-left)
        s3 = rearrange(x_val, 'b h w d -> b (w h) d')
        # Direction 4: col-major backward (bottom-left -> top-right)
        s4 = s3.flip(1)

        # ── Run SSM scan independently on each sequence ───────────────────────
        y1 = self.ssm_scan(s1)
        y2 = self.ssm_scan(s2).flip(1)                 # re-flip so tokens align
        y3 = self.ssm_scan(s3)
        y4 = self.ssm_scan(s4).flip(1)

        # ── Cross-Merge: rearrange column scans back to row-major and sum ─────
        y3 = rearrange(y3, 'b (w h) d -> b (h w) d', h=H, w=W)
        y4 = rearrange(y4, 'b (w h) d -> b (h w) d', h=H, w=W)

        y = y1 + y2 + y3 + y4                          # (B, H*W, d_inner)
        y = self.norm(y)
        y = rearrange(y, 'b (h w) d -> b h w d', h=H, w=W)

        # Gating
        y = y * self.act(z)

        # Project back to d_model
        y = self.out_proj(y)                            # (B, H, W, d_model)
        return y


# ─────────────────────────────────────────────────────────────────────────────
# 2. VSS Block: Vision State Space Block
#    Wraps SS2D in a standard pre-norm residual block (like a Transformer block
#    but replacing self-attention with SS2D).
# ─────────────────────────────────────────────────────────────────────────────

class VSSBlock(nn.Module):
    """
    Pre-norm Vision State Space Block.
    Input/output shape: (B, H, W, C)
    """
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 3,
                 expand: int = 2, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.ss2d = SS2D(dim, d_state=d_state, d_conv=d_conv, expand=expand)

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        x = x + self.ss2d(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 3. PatchEmbed: Initial patch embedding  (B,C,H,W) -> (B,H/4,W/4,embed_dim)
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """
    Overlapping patch embedding using two stacked 3×3 stride-2 convolutions.
    Gives receptive field of 7×7 while keeping the patch-to-token semantics.
    Converts (B, in_channels, H, W) -> (B, H/4, W/4, embed_dim)
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim // 2),
            nn.GELU(),
            nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                        # (B, embed_dim, H/4, W/4)
        x = rearrange(x, 'b c h w -> b h w c') # -> (B, H/4, W/4, embed_dim)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 4. PatchMerge: Downsampling between encoder stages
#    Halves H and W, doubles channels.
# ─────────────────────────────────────────────────────────────────────────────

class PatchMerge(nn.Module):
    """
    Merges 2×2 neighboring tokens and projects to 2×C dimensions.
    (B, H, W, C) -> (B, H/2, W/2, 2C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, C = x.shape
        # Concatenate 2×2 windows along the channel axis
        x0 = x[:, 0::2, 0::2, :]   # top-left
        x1 = x[:, 1::2, 0::2, :]   # bottom-left
        x2 = x[:, 0::2, 1::2, :]   # top-right
        x3 = x[:, 1::2, 1::2, :]   # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)    # (B, H/2, W/2, 4C)
        x = self.norm(x)
        x = self.reduction(x)                       # (B, H/2, W/2, 2C)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 5. MambaEncoderStage: One full encoder stage
#    N × VSSBlocks (no spatial change) + PatchMerge (optional, for downsampling)
# ─────────────────────────────────────────────────────────────────────────────

class MambaEncoderStage(nn.Module):
    """
    Runs `depth` VSSBlocks on the current resolution, then optionally downsamples.
    Returns (features_before_downsample, features_after_downsample).
    The 'before' tensor is what gets routed to the decoder as a skip connection.
    """
    def __init__(self, dim: int, depth: int, downsample: bool = True, **vss_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([VSSBlock(dim, **vss_kwargs) for _ in range(depth)])
        self.downsample = PatchMerge(dim) if downsample else None

    def forward(self, x: torch.Tensor):
        for blk in self.blocks:
            x = blk(x)
        skip = x
        if self.downsample is not None:
            x = self.downsample(x)
        return skip, x


# ─────────────────────────────────────────────────────────────────────────────
# 6. DecoderBlock: One upsampling stage with skip connection
# ─────────────────────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """
    Upsamples by 2x using transposed convolution, concatenates skip features,
    then refines with two 3×3 Conv-BN-ReLU layers.
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)                              # Upsample
        # Handle mismatches from odd input sizes
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)             # Concatenate skip
        x = self.conv(x)                            # Refine
        return x


# ─────────────────────────────────────────────────────────────────────────────
# 7. MambaSeg2D: The Full Model
# ─────────────────────────────────────────────────────────────────────────────

class MambaSeg2D(nn.Module):
    """
    2D Vision-Mamba Segmentation Network for Pancreatic CT.
    Plug-in replacement for the nnUNet trainer's build_network_architecture().

    Architecture overview:
        Input (B, 1, H, W)
            ↓ PatchEmbed     → (B, H/4, W/4, 96)        [stride-4 initial tokens]
            ↓ Stage 0        → (B, H/4, W/4, 96)         skip0, then merge
            ↓ Stage 1        → (B, H/8, W/8, 192)        skip1, then merge
            ↓ Stage 2        → (B, H/16, W/16, 384)      skip2, then merge
            ↓ Stage 3        → (B, H/32, W/32, 768)      skip3, then merge
            ↓ Bottleneck     → (B, H/32, W/32, 768)      (no spatial change)
            ↑ Decoder3       → (B, H/16, W/16, 384)      + skip2
            ↑ Decoder2       → (B, H/8, W/8, 192)        + skip1
            ↑ Decoder1       → (B, H/4, W/4, 96)         + skip0
            ↑ Decoder0       → (B, H, W, 48)             (upsample ×4 to original)
            ↓ Heads          → [(B, num_classes, H, W), (B, nc, H/2, W/2), (B, nc, H/4, W/4)]

    Deep supervision list is ordered from HIGHEST to LOWEST resolution,
    as expected by nnUNet's deep supervision loss wrapper.

    Args:
        in_channels     : Number of image channels (1 for CT)
        num_classes     : Number of segmentation classes (3 for BG/Pancreas/Tumor)
        embed_dim       : Base embedding dimension (channels at Stage 0)
        depths          : Number of VSSBlocks per encoder stage
        deep_supervision: If True (and model.training), returns list of logits
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 3,
        embed_dim: int = 96,
        depths: tuple = (2, 2, 6, 2),
        d_state: int = 16,
        deep_supervision: bool = True,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]  # [96, 192, 384, 768]

        # ── Encoder ──────────────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(in_channels=in_channels, embed_dim=dims[0])

        self.stages = nn.ModuleList()
        for i, (dim, depth) in enumerate(zip(dims, depths)):
            is_last = (i == len(dims) - 1)
            self.stages.append(
                MambaEncoderStage(
                    dim=dim,
                    depth=depth,
                    downsample=not is_last,     # last stage doesn't downsample
                    d_state=d_state,
                )
            )

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.ModuleList([VSSBlock(dims[-1], d_state=d_state) for _ in range(2)])

        # ── Decoder ──────────────────────────────────────────────────────────
        # PatchEmbed stride=4 means encoder starts at H/4.
        # Skip resolutions (before PatchMerge):
        #   skip[0]: H/4  x W/4  (96ch)   -- encoder stage 0
        #   skip[1]: H/8  x W/8  (192ch)  -- encoder stage 1
        #   skip[2]: H/16 x W/16 (384ch)  -- encoder stage 2
        # Bottleneck sits at H/32 x W/32   (768ch)
        #
        # Decoder stages (bottom-up), with DS head at every step:
        #   D3: H/32 -> H/16  + skip[2] -> 384ch  [1/16 head]
        #   D2: H/16 -> H/8   + skip[1] -> 192ch  [1/8  head]
        #   D1: H/8  -> H/4   + skip[0] -> 96ch   [1/4  head]
        #   D0a: H/4 -> H/2   (no skip)  -> 48ch  [1/2  head]
        #   D0b: H/2 -> H     (no skip)  -> 24ch  [1/1  head]
        self.decoders = nn.ModuleList([
            DecoderBlock(in_channels=dims[3], skip_channels=dims[2], out_channels=dims[2]),  # D3
            DecoderBlock(in_channels=dims[2], skip_channels=dims[1], out_channels=dims[1]),  # D2
            DecoderBlock(in_channels=dims[1], skip_channels=dims[0], out_channels=dims[0]),  # D1
        ])
        # D0a and D0b are two explicit upsample+refine steps (no skip, compensating PatchEmbed)
        ch_half  = dims[0] // 2   # 48
        ch_full  = dims[0] // 4   # 24
        self.d0a = nn.Sequential(
            nn.ConvTranspose2d(dims[0], ch_half, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_half), nn.ReLU(inplace=True),
            nn.Conv2d(ch_half, ch_half, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_half), nn.ReLU(inplace=True),
        )
        self.d0b = nn.Sequential(
            nn.ConvTranspose2d(ch_half, ch_full, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch_full), nn.ReLU(inplace=True),
            nn.Conv2d(ch_full, ch_full, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch_full), nn.ReLU(inplace=True),
        )

        # ── Deep Supervision Heads ───────────────────────────────────────────
        # 5 heads matching nnUNet's 5 downsampled ground-truth scales:
        #   [H, H/2, H/4, H/8, H/16]
        self.ds_heads = nn.ModuleList([
            nn.Conv2d(ch_full,   num_classes, kernel_size=1),  # H     (full)
            nn.Conv2d(ch_half,   num_classes, kernel_size=1),  # H/2
            nn.Conv2d(dims[0],   num_classes, kernel_size=1),  # H/4
            nn.Conv2d(dims[1],   num_classes, kernel_size=1),  # H/8
            nn.Conv2d(dims[2],   num_classes, kernel_size=1),  # H/16
        ])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        """
        x: (B, C, H, W)  — standard nnUNet input
        Returns:
            Training mode + deep_supervision=True:
                [H, H/2, H/4, H/8, H/16]  (5 outputs, highest to lowest resolution)
                This matches nnUNet's 5 ground-truth downsampled targets exactly.
            Eval / inference mode:
                logits at full resolution H x W
        """
        B, C, H, W = x.shape

        # ── Encoder ──────────────────────────────────────────────────────────
        x = self.patch_embed(x)                     # (B, H/4, W/4, embed_dim)

        skips = []
        for stage in self.stages:
            skip, x = stage(x)                      # skip: before merge, x: after merge
            skips.append(skip)

        # ── Bottleneck ───────────────────────────────────────────────────────
        for blk in self.bottleneck:
            x = blk(x)

        # Convert from (B, H, W, C) to (B, C, H, W) for CNN decoder
        x = rearrange(x, 'b h w c -> b c h w')
        skips_2d = [rearrange(s, 'b h w c -> b c h w') for s in skips]

        # ── Decoder (bottom-up) ───────────────────────────────────────────────
        x = self.decoders[0](x, skips_2d[2])        # H/32 -> H/16 + skip[2]
        logits_16 = self.ds_heads[4](x)              # DS head at H/16

        x = self.decoders[1](x, skips_2d[1])        # H/16 -> H/8  + skip[1]
        logits_8  = self.ds_heads[3](x)              # DS head at H/8

        x = self.decoders[2](x, skips_2d[0])        # H/8  -> H/4  + skip[0]
        logits_4  = self.ds_heads[2](x)              # DS head at H/4

        x = self.d0a(x)                              # H/4 -> H/2
        logits_2  = self.ds_heads[1](x)              # DS head at H/2

        x = self.d0b(x)                              # H/2 -> H (full resolution)
        logits_1  = self.ds_heads[0](x)              # DS head at H

        if self.deep_supervision and self.training:
            # nnUNet expects: [best_resolution, ..., worst_resolution]
            return [logits_1, logits_2, logits_4, logits_8, logits_16]
        else:
            return logits_1


# ─────────────────────────────────────────────────────────────────────────────
# Quick Sanity Check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    model = MambaSeg2D(in_channels=1, num_classes=3, embed_dim=96, depths=(2, 2, 6, 2))
    model.train()

    dummy_input = torch.randn(2, 1, 256, 256)   # Batch=2, CT=1ch, 256×256 patch
    outputs = model(dummy_input)

    print("=== MambaSeg2D Training Forward Pass ===")
    for i, o in enumerate(outputs):
        print(f"  Deep supervision output [{i}]: {tuple(o.shape)}")

    model.eval()
    with torch.no_grad():
        out = model(dummy_input)
    print(f"\n=== MambaSeg2D Inference Output ===")
    print(f"  Single output (inference): {tuple(out.shape)}")
