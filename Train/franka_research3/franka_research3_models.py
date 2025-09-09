# model.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from franka_research3_utils import (
    MODEL_NAME, NUM_ANGLES, NUM_JOINTS, HEATMAP_SIZE, FEATURE_DIM
)

# ==== Torch FK (Modified DH, differentiable) ====
class FrankaFK(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        a     = torch.tensor([0.0, 0.0, 0.0, 0.0825, -0.0825, 0.0, 0.088, 0.0], device=device)
        d     = torch.tensor([0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.0, 0.107], device=device)
        alpha = torch.tensor([0.0, -90.0, 90.0, 90.0, -90.0, 90.0, 90.0, 0.0], device=device)
        self.register_buffer('a', a); self.register_buffer('d', d); self.register_buffer('alpha', alpha)

    def _mdh(self, a_s, d_s, alpha_deg_s, theta_deg_b):  # *_s: scalar(0-d) 또는 1-d, theta_deg_b: [B]
        # 스칼라들을 배치 크기로 확장
        B = theta_deg_b.shape[0]
        a     = a_s.expand(B)
        d     = d_s.expand(B)
        alpha = alpha_deg_s.expand(B)

        alpha = torch.deg2rad(alpha)
        th    = torch.deg2rad(theta_deg_b)
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        ct, st = torch.cos(th),    torch.sin(th)

        zeros = torch.zeros_like(ct); ones = torch.ones_like(ct)

        row0 = torch.stack([ct, -st, zeros, a], dim=-1)                 # [B,4]
        row1 = torch.stack([st*ca, ct*ca, -sa, -d*sa], dim=-1)
        row2 = torch.stack([st*sa, ct*sa,  ca,  d*ca], dim=-1)
        row3 = torch.stack([zeros, zeros, zeros, ones], dim=-1)
        return torch.stack([row0, row1, row2, row3], dim=-2)            # [B,4,4]

    def forward(self, joint_deg):  # [B,7] 또는 [B,8]
        B, A = joint_deg.shape
        if A < 8:
            pad = torch.zeros(B, 8-A, device=joint_deg.device, dtype=joint_deg.dtype)
            theta_all = torch.cat([joint_deg, pad], dim=1)
        else:
            theta_all = joint_deg

        T = torch.eye(4, device=joint_deg.device).unsqueeze(0).repeat(B,1,1)
        pts = [T[..., :3, 3]]  # base

        for i in range(8):
            Ti = self._mdh(self.a[i], self.d[i], self.alpha[i], theta_all[:, i])  # [B,4,4]
            T = T @ Ti
            pts.append(T[..., :3, 3])

        return torch.stack(pts, dim=1)  # [B, 9, 3]



# -----------------------------
# Backbone
# -----------------------------
class DINOv3Backbone(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, image_tensor_batch):
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor_batch)
        tokens = outputs.last_hidden_state
        num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
        if not hasattr(self, "_logged_reg"):
            print(f"[DINOv3Backbone] num_register_tokens = {num_reg}, total tokens = {tokens.shape[1]}")
            self._logged_reg = True
        # CLS + REG 제외 → 패치 토큰만
        patch_tokens = tokens[:, 1 + num_reg :, :]   # (B, N_patches, D)
        return patch_tokens

# -----------------------------
# Multi-View Fusion (token-level)
# -----------------------------
class MultiViewFusion(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, num_heads=8, dropout=0.1, num_queries=16, num_layers=2):
        super().__init__()
        self.global_queries = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim*4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)

    def forward(self, view_features_list):
        if len(view_features_list) == 0:
            raise ValueError("MultiViewFusion: empty view feature list")
        memory = torch.cat(view_features_list, dim=1)      # (B, sum(Np_i), D)
        b = memory.size(0)
        queries = self.global_queries.repeat(b, 1, 1)      # (B, Q, D)
        return self.decoder(tgt=queries, memory=memory)     # (B, Q, D)

# -----------------------------
# Heads (Keypoint)
# -----------------------------
class LightCNNStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(),
            nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU()
        )

    def forward(self, x):
        f4 = self.conv_block1(x)  # 1/4
        f8 = self.conv_block2(f4) # 1/8
        return f4, f8

class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels)
        )
        self.residual = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        y = self.proj(x)
        y = self.refine(y) + self.residual(x)
        return F.gelu(y)

class FusedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels+skip_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.GELU()
        )

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))

class UNetViTKeypointHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_joints=NUM_JOINTS, heatmap_size=(128, 128)):
        super().__init__()
        self.heatmap_size = heatmap_size
        self.token_fuser = TokenFuser(input_dim, 256)
        self.dec1 = FusedUpsampleBlock(256, 64, 128)
        self.dec2 = FusedUpsampleBlock(128, 32, 64)
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.pred = nn.Conv2d(64, num_joints, 3, 1, 1)

    def forward(self, dino_features, cnn_features):
        f4, f8 = cnn_features
        b, n, d = dino_features.shape
        h = w = int(n ** 0.5)
        assert h * w == n, f"[UNetViTKeypointHead] patch tokens must form a square grid, got n={n}."
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        x = self.token_fuser(x)
        x = self.dec1(x, f8)
        x = self.dec2(x, f4)
        x = self.up_final(x)
        heat = self.pred(x)  # (B,J,h,w) upsampled to near input stride
        return F.interpolate(heat, size=self.heatmap_size, mode='bilinear', align_corners=False)

# -----------------------------
# Soft-argmax & KP token encoder
# -----------------------------
class SoftArgmax2D(nn.Module):
    def __init__(self, H=128, W=128, beta=1.0, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps
        xs = torch.linspace(0, W-1, W)
        ys = torch.linspace(0, H-1, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x)  # (H,W)
        self.register_buffer('grid_y', grid_y)  # (H,W)

    def forward(self, heat):  # (B, J, H, W)
        B,J,H,W = heat.shape
        h = heat.view(B*J, H*W)
        p = F.softmax(self.beta * h, dim=-1) + self.eps  # (B*J, HW)
        ex = torch.sum(p * self.grid_x.flatten(), dim=-1)  # (B*J,)
        ey = torch.sum(p * self.grid_y.flatten(), dim=-1)
        ex = ex.view(B, J); ey = ey.view(B, J)
        ent = -torch.sum(p * p.log(), dim=-1).view(B, J)  # (B,J) entropy
        return ex, ey, ent

class KeypointTokenEncoder(nn.Module):
    def __init__(self, num_views, num_joints, embed_dim, use_uncert=True):
        super().__init__()
        in_dim = 3 if use_uncert else 2
        self.view_embed  = nn.Embedding(num_views,  embed_dim)
        self.joint_embed = nn.Embedding(num_joints, embed_dim)   # 추가
        self.mlp = nn.Sequential(
            nn.Linear(in_dim + embed_dim*2, embed_dim),  # view + joint
            nn.GELU(), nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim)
        )
        self.num_joints = num_joints

    def forward(self, xyu_per_view, view_idx_per_view):
        feats = []
        for (feat, vidx) in zip(xyu_per_view, view_idx_per_view):
            B,J,C = feat.shape
            ve = self.view_embed.weight[vidx].expand(B, J, -1)             # (B,J,D)
            je = self.joint_embed.weight.view(1, J, -1).expand(B, J, -1)   # (B,J,D)
            inp = torch.cat([feat, ve, je], dim=-1)                         # (B,J,C+2D)
            feats.append(self.mlp(inp))
        return torch.cat(feats, dim=1)  # (B, V*J, D)


# (model.py 상단 import와 클래스들 사이 적당한 곳에 추가)
class CNNTokenEncoder(nn.Module):
    """
    (f4, f8) CNN feature → 소수의 토큰으로 요약해 D차원으로 투영
    tokens_per_view = s*s 로 만들고, AdaptiveAvgPool로 s×s 그리드 토큰 생성
    """
    def __init__(self, in_ch_f4=32, in_ch_f8=64, embed_dim=FEATURE_DIM, tokens_per_view: int = 16):
        super().__init__()
        self.s = int(tokens_per_view ** 0.5)
        assert self.s * self.s == tokens_per_view, "tokens_per_view must be a square (e.g., 4, 9, 16, ...)"

        # f8: (B,64,h8,w8) → D
        self.proj_f8 = nn.Conv2d(in_ch_f8, embed_dim, kernel_size=1, bias=False)
        # f4: (B,32,h4,w4) → stride-2 다운샘플 → D
        self.down_f4 = nn.Conv2d(in_ch_f4, in_ch_f4, kernel_size=3, stride=2, padding=1, bias=False)
        self.proj_f4 = nn.Conv2d(in_ch_f4, embed_dim, kernel_size=1, bias=False)

        self.pool = nn.AdaptiveAvgPool2d((self.s, self.s))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, f4, f8):
        # f8 branch
        t8 = self.proj_f8(f8)                 # (B,D,h8,w8)
        t8 = self.pool(t8)                    # (B,D,s,s)

        # f4 branch (downsample to f8 scale then project)
        f4d = self.down_f4(f4)                # (B,32,h8,w8) roughly
        t4 = self.proj_f4(f4d)                # (B,D,h8,w8)
        t4 = self.pool(t4)                    # (B,D,s,s)

        t = 0.5 * (t8 + t4)                   # (B,D,s,s)
        B, D, S, _ = t.shape
        t = t.permute(0,2,3,1).contiguous().view(B, S*S, D)  # (B, tokens_per_view, D)
        t = self.norm(t)
        return t

# -----------------------------
# Angle Head (memory-driven)
# -----------------------------
class JointAngleHead(nn.Module):
    """
    외부 memory(=fused + kp_tokens)를 입력 받아 각도(sin,cos) 회귀
    """
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES,
                 num_queries=16, nhead=8, num_layers=2):
        super().__init__()
        self.num_angles = num_angles
        self.pose_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        layer = nn.TransformerDecoderLayer(
            d_model=input_dim, nhead=nhead, dim_feedforward=input_dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.mlp = nn.Sequential(
            nn.LayerNorm(input_dim*num_queries),
            nn.Linear(input_dim*num_queries, 512), nn.GELU(), nn.LayerNorm(512),
            nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, num_angles * 2)
        )

    def forward(self, memory):
        """
        memory: (B, T, D)  ← fused tokens (+ optional kp tokens)
        returns: (B, A, 2) with unit norm
        """
        B = memory.size(0)
        q = self.pose_queries.repeat(B, 1, 1)              # (B,Q,D)
        attn = self.decoder(tgt=q, memory=memory)          # (B,Q,D)
        out = self.mlp(attn.flatten(start_dim=1))          # (B, 2*A)
        out = out.view(B, self.num_angles, 2)
        out = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return out

# -----------------------------
# Final Model
# -----------------------------
class DINOv3PoseEstimator(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_joints=NUM_JOINTS, num_angles=NUM_ANGLES,
                 known_view_keys=None, max_views=10):
        super().__init__()
        self.backbone = DINOv3Backbone(model_name)
        feature_dim = self.backbone.model.config.hidden_size

        # view embedding
        if known_view_keys is not None:
            self.known_view_keys = list(known_view_keys)
            self.view_to_idx = {k: i for i, k in enumerate(self.known_view_keys)}
            self.view_embeddings = nn.Embedding(len(self.known_view_keys), feature_dim)
            self.num_known_views = len(self.known_view_keys)
        else:
            self.known_view_keys = None
            self.view_to_idx = {}
            self.view_embeddings = nn.Embedding(max_views, feature_dim)
            self.num_known_views = max_views

        # branches
        self.cnn_stem = LightCNNStem()
        self.fusion   = MultiViewFusion(feature_dim=feature_dim)
        self.kpt_head = UNetViTKeypointHead(input_dim=feature_dim, num_joints=num_joints)
        self.kpt_enricher = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=8, dim_feedforward=feature_dim*4,
            dropout=0.1, activation='gelu', batch_first=True
        )
        self.ang_head = JointAngleHead(input_dim=feature_dim, num_angles=num_angles)

        # kp → tokens
        self.softarg = SoftArgmax2D(H=HEATMAP_SIZE[0], W=HEATMAP_SIZE[1], beta=1.0)
        self.kp_token_enc = KeypointTokenEncoder(
            num_views=self.num_known_views, num_joints=num_joints,
            embed_dim=feature_dim, use_uncert=True
        )
        # 기존 __init__ 말미에 추가
        self.cnn_token_enc = CNNTokenEncoder(in_ch_f4=32, in_ch_f8=64,
                                            embed_dim=feature_dim, tokens_per_view=16)
        self.detach_cnn = False         # 필요시 True로 켜서 angle 쪽만 먼저 안정화
        self.cnn_token_dropout = 0.1    # 0.0~0.3 권장 (학습 안정/일반화)


        # 처음엔 detach ON 권장 (훈련 루프에서 set_detach_kp(False)로 끔)
        self.detach_kp = False

    def set_detach_kp(self, flag: bool):
        self.detach_kp = bool(flag)
        
    def set_detach_cnn(self, flag: bool):
        self.detach_cnn = bool(flag)


    def forward(self, multi_view_images: dict):
        all_dino = []
        all_cnn  = {}
        ordered = list(multi_view_images.keys())

        view_indices = []
        # --- NEW: cnn 토큰 모을 리스트
        cnn_token_list = []

        for k in ordered:
            x = multi_view_images[k]                       # (B,3,H,W)
            dino = self.backbone(x)                        # (B,Np,D)

            # view index
            if self.known_view_keys is not None:
                if k not in self.view_to_idx:
                    continue
                idx = self.view_to_idx[k]
            else:
                if k not in self.view_to_idx:
                    cur = len(self.view_to_idx)
                    if cur >= self.view_embeddings.num_embeddings:
                        raise ValueError("Exceeded maximum number of views.")
                    self.view_to_idx[k] = cur
                idx = self.view_to_idx[k]

            # add view bias to ViT tokens
            emb = self.view_embeddings(torch.tensor([idx], device=dino.device)).unsqueeze(0)  # (1,1,D)
            all_dino.append(dino + emb)

            # CNN stem
            f4, f8 = self.cnn_stem(x)
            all_cnn[k] = (f4, f8)
            view_indices.append(idx)

            # --- NEW: CNN tokens (B, Tc, D) + view embedding 주입
            cnn_tokens = self.cnn_token_enc(f4, f8)        # (B, Tc, D)
            ve = self.view_embeddings(torch.tensor([idx], device=cnn_tokens.device)).view(1,1,-1)
            cnn_tokens = cnn_tokens + ve                   # 뷰 구분 신호
            cnn_token_list.append(cnn_tokens)

        if len(all_dino) == 0:
            raise ValueError("DINOv3PoseEstimator: no valid views in the batch.")

        # (1) fuse views
        fused = self.fusion(all_dino)                      # (B,Q,D)

        # (2) keypoint heatmaps per view
        pred_hm_dict = {}
        kp_feat_per_view = []
        for i, k in enumerate(ordered[:len(all_dino)]):
            f4, f8 = all_cnn[k]
            enr = self.kpt_enricher(tgt=all_dino[i], memory=fused)  # (B,Np,D)
            hm  = self.kpt_head(enr, (f4, f8))                      # (B,J,H,W)
            pred_hm_dict[k] = hm

            # soft-argmax & uncertainty
            ex, ey, ent = self.softarg(hm)
            H, W = hm.shape[-2:]
            xyu = torch.stack([
                ex / max(1, (W-1)),
                ey / max(1, (H-1)),
                ent / (math.log(max(1, H*W)) + 1e-6)
            ], dim=-1)   # (B,J,3)
            kp_feat_per_view.append(xyu)

        # (3) kp tokens
        if len(kp_feat_per_view) > 0:
            kp_tokens = self.kp_token_enc(kp_feat_per_view, view_indices)  # (B,V*J,D)
            if self.detach_kp:
                kp_tokens = kp_tokens.detach()
        else:
            kp_tokens = None

        # --- NEW: cnn tokens 결합
        if len(cnn_token_list) > 0:
            cnn_tokens_all = torch.cat(cnn_token_list, dim=1)  # (B, V*Tc, D)
            if self.detach_cnn:
                cnn_tokens_all = cnn_tokens_all.detach()
            if self.training and self.cnn_token_dropout > 0:
                # 간단한 토큰 드롭아웃
                drop_prob = self.cnn_token_dropout
                B, T, D = cnn_tokens_all.shape
                mask = (torch.rand(B, T, 1, device=cnn_tokens_all.device) > drop_prob).float()
                cnn_tokens_all = cnn_tokens_all * mask
        else:
            cnn_tokens_all = None

        # (4) memory 확장: fused (+ kp) (+ cnn)
        memory_ext = fused
        if kp_tokens is not None:
            memory_ext = torch.cat([memory_ext, kp_tokens], dim=1)
        if cnn_tokens_all is not None:
            memory_ext = torch.cat([memory_ext, cnn_tokens_all], dim=1)

        # (5) angle head
        pred_angles = self.ang_head(memory_ext)            # (B,A,2)
        return pred_hm_dict, pred_angles
