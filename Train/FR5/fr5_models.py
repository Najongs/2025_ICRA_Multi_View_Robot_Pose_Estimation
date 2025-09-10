# fr5_models.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from fr5_utils import (
    MODEL_NAME, NUM_ANGLES, NUM_JOINTS, HEATMAP_SIZE, FEATURE_DIM
)

# -----------------------------
# Backbone & Fusion Modules
# -----------------------------
class DINOv3Backbone(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)

    def forward(self, image_tensor_batch):
        # ▼▼▼ [핵심 수정] with torch.no_grad()를 다시 추가하여 백본을 동결합니다. ▼▼▼
        with torch.no_grad():
            outputs = self.model(pixel_values=image_tensor_batch)
        
        tokens = outputs.last_hidden_state
        num_reg = int(getattr(self.model.config, "num_register_tokens", 0))
        if not hasattr(self, "_logged_reg"):
            print(f"[DINOv3Backbone] num_register_tokens = {num_reg}, total tokens = {tokens.shape[1]}")
            self._logged_reg = True
        
        patch_tokens = tokens[:, 1 + num_reg :, :]  # (B, N_patches, D)
        return patch_tokens

class MultiViewFusion(nn.Module):
    def __init__(self, feature_dim=FEATURE_DIM, num_heads=8, dropout=0.1, num_queries=16, num_layers=2):
        super().__init__()
        self.global_queries = nn.Parameter(torch.randn(1, num_queries, feature_dim))
        layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=num_heads, dim_feedforward=feature_dim*4, dropout=dropout, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
    def forward(self, view_features_list):
        if not view_features_list: raise ValueError("MultiViewFusion: empty view feature list")
        memory = torch.cat(view_features_list, dim=1)
        b = memory.size(0)
        queries = self.global_queries.repeat(b, 1, 1)
        return self.decoder(tgt=queries, memory=memory)

# -----------------------------
# Keypoint Head Modules
# -----------------------------
class LightCNNStem(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1, bias=False), nn.BatchNorm2d(16), nn.GELU(), nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.GELU())
        self.conv_block2 = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.GELU())
    def forward(self, x):
        return self.conv_block1(x), self.conv_block2(self.conv_block1(x))

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
        x = dino_features.permute(0, 2, 1).reshape(b, d, h, w)
        x = self.token_fuser(x)
        x = self.dec1(x, f8)
        x = self.dec2(x, f4)
        x = self.up_final(x)
        heat = self.pred(x)
        return F.interpolate(heat, size=self.heatmap_size, mode='bilinear', align_corners=False)

# (TokenFuser, FusedUpsampleBlock 정의는 생략 - 변경 없음)
class TokenFuser(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1)
        self.refine = nn.Sequential(nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels))
        self.residual = nn.Conv2d(in_channels, out_channels, 1)
    def forward(self, x):
        y = self.proj(x)
        y = self.refine(y) + self.residual(x)
        return F.gelu(y)

class FusedUpsampleBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.refine = nn.Sequential(nn.Conv2d(in_channels+skip_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU(), nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False), nn.BatchNorm2d(out_channels), nn.GELU())
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return self.refine(torch.cat([x, skip], dim=1))

# -----------------------------
# Tokenizer Modules
# -----------------------------
class SoftArgmax2D(nn.Module):
    def __init__(self, H=128, W=128, beta=1.0, eps=1e-6):
        super().__init__()
        self.beta = beta
        self.eps = eps
        xs = torch.linspace(0, W-1, W)
        ys = torch.linspace(0, H-1, H)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')
        self.register_buffer('grid_x', grid_x)
        self.register_buffer('grid_y', grid_y)
    def forward(self, heat):
        B,J,H,W = heat.shape
        p = F.softmax(self.beta * heat.view(B*J, H*W), dim=-1) + self.eps
        ex = torch.sum(p * self.grid_x.flatten(), dim=-1).view(B, J)
        ey = torch.sum(p * self.grid_y.flatten(), dim=-1).view(B, J)
        ent = -torch.sum(p * p.log(), dim=-1).view(B, J)
        return ex, ey, ent

class KeypointTokenEncoder(nn.Module):
    def __init__(self, num_views, num_joints, embed_dim, use_uncert=True):
        super().__init__()
        in_dim = 3 if use_uncert else 2
        self.view_embed  = nn.Embedding(num_views, embed_dim)
        self.joint_embed = nn.Embedding(num_joints, embed_dim)
        self.mlp = nn.Sequential(nn.Linear(in_dim + embed_dim*2, embed_dim), nn.GELU(), nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim))
    def forward(self, xyu_per_view, view_idx_per_view):
        feats = []
        for (feat, vidx) in zip(xyu_per_view, view_idx_per_view):
            B,J,_ = feat.shape
            ve = self.view_embed.weight[vidx].expand(B, J, -1)
            je = self.joint_embed.weight.unsqueeze(0).expand(B, -1, -1)
            inp = torch.cat([feat, ve, je], dim=-1)
            feats.append(self.mlp(inp))
        return torch.cat(feats, dim=1)

class CNNTokenEncoder(nn.Module):
    def __init__(self, in_ch_f4=32, in_ch_f8=64, embed_dim=FEATURE_DIM, tokens_per_view: int = 16):
        super().__init__()
        s = int(tokens_per_view ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d((s, s))
        self.proj_f8 = nn.Conv2d(in_ch_f8, embed_dim, 1, bias=False)
        self.down_f4 = nn.Conv2d(in_ch_f4, in_ch_f4, 3, 2, 1, bias=False)
        self.proj_f4 = nn.Conv2d(in_ch_f4, embed_dim, 1, bias=False)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, f4, f8):
        t8 = self.pool(self.proj_f8(f8))
        t4 = self.pool(self.proj_f4(self.down_f4(f4)))
        t = (t8 + t4) * 0.5
        B, D, S, _ = t.shape
        t = t.permute(0,2,3,1).contiguous().view(B, S*S, D)
        return self.norm(t)

# -----------------------------
# Angle Head
# -----------------------------
class JointAngleHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, num_angles=NUM_ANGLES, num_queries=16, nhead=8, num_layers=2):
        super().__init__()
        self.num_angles = num_angles
        self.pose_queries = nn.Parameter(torch.randn(1, num_queries, input_dim))
        layer = nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=input_dim*4, dropout=0.1, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim*num_queries), nn.Linear(input_dim*num_queries, 512), nn.GELU(), nn.LayerNorm(512), nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256), nn.Linear(256, num_angles * 2))
    def forward(self, memory):
        B = memory.size(0)
        q = self.pose_queries.repeat(B, 1, 1)
        attn = self.decoder(tgt=q, memory=memory)
        out = self.mlp(attn.flatten(start_dim=1)).view(B, self.num_angles, 2)
        return F.normalize(out, dim=-1)

# ==============================================================================
# Final Model (수정된 버전)
# ==============================================================================
class DINOv3PoseEstimator(nn.Module):
    # ▼▼▼ [핵심 수정] __init__에서 known_view_keys 제거, max_views를 기본으로 사용 ▼▼▼
    def __init__(self, model_name=MODEL_NAME, num_joints=NUM_JOINTS, num_angles=NUM_ANGLES, max_views=10):
        super().__init__()
        self.backbone = DINOv3Backbone(model_name)
        feature_dim = self.backbone.model.config.hidden_size

        # View Embedding: 더 이상 특정 키에 의존하지 않고, 최대 뷰 개수만 가정합니다.
        self.max_views = max_views
        self.view_embeddings = nn.Embedding(self.max_views, feature_dim)

        # branches
        self.cnn_stem = LightCNNStem()
        self.fusion   = MultiViewFusion(feature_dim=feature_dim)
        self.kpt_head = UNetViTKeypointHead(input_dim=feature_dim, num_joints=num_joints)
        self.kpt_enricher = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=8, dim_feedforward=feature_dim*4, dropout=0.1, activation='gelu', batch_first=True)
        self.ang_head = JointAngleHead(input_dim=feature_dim, num_angles=num_angles)

        # Tokenizers
        self.softarg = SoftArgmax2D(H=HEATMAP_SIZE[0], W=HEATMAP_SIZE[1], beta=1.0)
        self.kp_token_enc = KeypointTokenEncoder(
            num_views=self.max_views, # num_known_views 대신 max_views 사용
            num_joints=num_joints,
            embed_dim=feature_dim, use_uncert=True
        )
        self.cnn_token_enc = CNNTokenEncoder(embed_dim=feature_dim)
        
        # 학습 제어용 플래그
        self.detach_kp = False
        self.detach_cnn = False
        self.drop_prob_scheduled = 0.0 # 학습 루프에서 직접 제어

    def set_detach_kp(self, flag: bool): self.detach_kp = bool(flag)
    def set_detach_cnn(self, flag: bool): self.detach_cnn = bool(flag)

    # ▼▼▼ [핵심 수정] forward에서 입력 딕셔너리의 순서를 이용해 view index를 동적으로 할당 ▼▼▼
    def forward(self, multi_view_images: dict):
        all_dino, all_cnn, cnn_token_list, view_indices = [], {}, [], []
        
        ordered_keys = list(multi_view_images.keys())
        if len(ordered_keys) > self.max_views:
            ordered_keys = ordered_keys[:self.max_views]

        for i, k in enumerate(ordered_keys):
            x = multi_view_images[k]
            dino = self.backbone(x)

            # View Embedding: i번째 뷰로 임베딩 적용
            ve = self.view_embeddings.weight[i].view(1,1,-1)
            all_dino.append(dino + ve)
            
            f4, f8 = self.cnn_stem(x)
            all_cnn[k] = (f4, f8)
            view_indices.append(i)
            
            cnn_tokens = self.cnn_token_enc(f4, f8)
            cnn_token_list.append(cnn_tokens + ve)

        if not all_dino: raise ValueError("DINOv3PoseEstimator: no valid views in the batch.")

        fused = self.fusion(all_dino)

        pred_hm_dict, kp_feat_per_view = {}, []
        for i, k in enumerate(ordered_keys):
            enr = self.kpt_enricher(tgt=all_dino[i], memory=fused)
            hm  = self.kpt_head(enr, all_cnn[k])
            pred_hm_dict[k] = hm

            ex, ey, ent = self.softarg(hm)
            H, W = hm.shape[-2:]
            xyu = torch.stack([ex / (W-1+1e-6), ey / (H-1+1e-6), ent / (math.log(H*W)+1e-6)], dim=-1)
            kp_feat_per_view.append(xyu)
            
        kp_tokens = self.kp_token_enc(kp_feat_per_view, view_indices)
        if self.detach_kp: kp_tokens = kp_tokens.detach()

        cnn_tokens_all = torch.cat(cnn_token_list, dim=1)
        if self.detach_cnn: cnn_tokens_all = cnn_tokens_all.detach()
        if self.training and self.drop_prob_scheduled > 0:
            B, T, D = cnn_tokens_all.shape
            mask = (torch.rand(B, T, 1, device=cnn_tokens_all.device) > self.drop_prob_scheduled).float()
            cnn_tokens_all = cnn_tokens_all * mask

        memory_ext = torch.cat([fused, kp_tokens, cnn_tokens_all], dim=1)
        pred_angles = self.ang_head(memory_ext)
        
        return pred_hm_dict, pred_angles