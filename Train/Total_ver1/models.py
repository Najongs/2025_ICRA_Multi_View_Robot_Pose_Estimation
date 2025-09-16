import math
from typing import Dict, Optional, Tuple, Union, Literal
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel

from dataclasses import dataclass
from typing import Tuple, Optional, Literal
from contextlib import nullcontext

@dataclass
class ModelCfg:
    MODEL_NAME: str
    NUM_ANGLES: int
    NUM_JOINTS: int
    FEATURE_DIM: int = 768
    HEATMAP_SIZE: Tuple[int, int] = (128, 128)
    MAX_VIEWS_PER_GROUP: int = 8

    # fusion / middle / token-level defaults
    FUSION: Literal["auto","late","middle","early"] = "auto"
    DEFAULT_FUSION_FOR_MULTI: Literal["late","middle","early"] = "late"
    FREEZE_BACKBONE: bool = True

    MIDDLE_HEADS: int = 4
    MIDDLE_DS: int = 2
    MIDDLE_LAMBDA_EPI: float = 0.05
    MIDDLE_TEMPERATURE: float = 1.0
    MIDDLE_NUM_VIEW_PROTOTYPES: int = 8
    EARLY_REDUCE_DIM: Optional[int] = None

    TOKEN_NUM_QUERIES: int = 16
    TOKEN_NUM_HEADS: int = 8
    TOKEN_NUM_LAYERS: int = 2
    USE_AUTO_VIEW_FOR_TOKENS: bool = True

MODEL_VIT  = "facebook/dinov3-vitl16-pretrain-lvd1689m"
MODEL_CNX  = "facebook/dinov3-convnext-large-pretrain-lvd1689m"

# AutoViewEmbed

class Pose3DHead(nn.Module):
    """ 글로벌 latent (B,Q,D) -> 3D 관절 (B,J,3) """
    def __init__(self, dim:int, num_joints:int, num_queries:int=16):
        super().__init__()
        self.j = num_joints
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim*num_queries),
            nn.Linear(dim*num_queries, dim),
            nn.GELU(),
            nn.Linear(dim, num_joints*3)
        )
    def forward(self, latent: torch.Tensor):
        B,Q,D = latent.shape
        x = latent.reshape(B, Q*D)
        xyz = self.mlp(x).view(B, self.j, 3)
        return xyz

def project_points(Pw: torch.Tensor, Ks: torch.Tensor, Rts: torch.Tensor, eps: float=1e-6):
    """
    Pw:  (B,J,3)   # world coords
    Ks:  (B,V,3,3)
    Rts: (B,V,3,4) # [R|t], x_cam = R * x_w + t
    return: (B,V,J,2)
    """
    B,J,_ = Pw.shape
    V = Ks.shape[1]
    ones = torch.ones(B, J, 1, device=Pw.device, dtype=Pw.dtype)
    Pw_h = torch.cat([Pw, ones], dim=-1)                             # (B,J,4)

    # (B,V,3,4) x (B,1,J,4,1) → (B,V,J,3,1)
    Pw_h1 = Pw_h.unsqueeze(1).unsqueeze(-1)                          # (B,1,J,4,1)
    cam = torch.matmul(Rts.unsqueeze(2), Pw_h1).squeeze(-1)          # (B,V,J,3)
    # K * cam
    uvw = torch.einsum("bvcd,bvjd->bvjc", Ks, cam)                   # (B,V,J,3)
    u = uvw[...,0] / (uvw[...,2].clamp_min(eps))
    v = uvw[...,1] / (uvw[...,2].clamp_min(eps))
    return torch.stack([u, v], dim=-1)                               # (B,V,J,2)

class TriangulationConsistencyLoss(nn.Module):
    """
    재투영(3D)과 히트맵 소프트아르그맥스(2D) 간 일관성 로스
    - 옵션: 신뢰도(conf) 가중
    """
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
    def forward(self, proj_2d: torch.Tensor, heat_2d: torch.Tensor, conf: Optional[torch.Tensor]=None):
        # (B,V,J,2), (B,V,J,2), (B,V,J)
        diff = torch.abs(proj_2d - heat_2d)                           # L1
        if conf is not None:
            w = conf.detach().unsqueeze(-1).clamp_min(1e-3)           # (B,V,J,1)
            diff = diff * w
        if self.reduction == "mean":
            return diff.mean()
        elif self.reduction == "sum":
            return diff.sum()
        else:
            return diff


class CameraPE(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3*3 + 3*4, dim//2), nn.GELU(),
            nn.Linear(dim//2, dim)
        )
    def forward(self, K: torch.Tensor, Rt: torch.Tensor):
        if K.dim()==3: K = K.unsqueeze(0).expand(Rt.shape[0], -1, -1, -1)
        if Rt.dim()==3: Rt = Rt.unsqueeze(0).expand(K.shape[0], -1, -1, -1)
        B,V = K.shape[:2]
        x = torch.cat([K.reshape(B,V,-1), Rt.reshape(B,V,-1)], dim=-1)
        return self.mlp(x)  # (B,V,dim)

class AutoViewEmbed(nn.Module):
    """
    카메라 파라미터(K,[R|t])로부터 뷰 임베딩을 자동 학습.
    descriptor -> routing -> soft prototypes (K개) -> weighted sum
    """
    def __init__(self, dim:int, num_prototypes:int=8, hidden:int=None, temperature:float=1.0):
        super().__init__()
        self.Kp = num_prototypes
        hid = hidden or max(128, dim//2)
        self.router = nn.Sequential(
            nn.Linear(16, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, self.Kp)
        )
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(self.Kp, dim) * 0.02)

    @staticmethod
    def _cam_center(Rt):
        R = Rt[..., :3, :3]
        t = Rt[..., :3, 3]
        return -(R.transpose(-1,-2) @ t.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def _angles(R):
        z = R[..., 2, :]
        az = torch.atan2(z[..., 0], z[..., 2])
        el = torch.asin(torch.clamp(z[..., 1], -1, 1))
        return az, el

    @staticmethod
    def _fov_proxy(K):
        fx = K[..., 0, 0]; fy = K[..., 1, 1]
        cx = K[..., 0, 2]; cy = K[..., 1, 2]
        invfx = 1.0/(fx+1e-6); invfy = 1.0/(fy+1e-6)
        return invfx, invfy, cx, cy

    def _build_desc(self, K, Rt, target=None):
        if K.dim()==3: K = K.unsqueeze(0).expand(Rt.shape[0], -1, -1, -1)
        if Rt.dim()==3: Rt = Rt.unsqueeze(0).expand(K.shape[0], -1, -1, -1)
        B,V = K.shape[:2]
        R = Rt[..., :3, :3]
        C = self._cam_center(Rt)              # (B,V,3)
        az, el = self._angles(R)              # (B,V)
        invfx, invfy, cx, cy = self._fov_proxy(K)
        if target is None:
            target = torch.zeros(B,3, device=Rt.device, dtype=Rt.dtype)
        baseline = C - target.unsqueeze(1)     # (B,V,3)
        dist = torch.linalg.norm(baseline, dim=-1)

        feat = [
            torch.sin(az), torch.cos(az),
            torch.sin(el), torch.cos(el),
            invfx, invfy, cx, cy,
            baseline[...,0], baseline[...,1], baseline[...,2],
            dist, torch.ones_like(dist)
        ]
        desc = torch.stack(feat, dim=-1)      # (B,V,16)
        desc = (desc - desc.mean(dim=(0,1), keepdim=True)) / (desc.std(dim=(0,1), keepdim=True) + 1e-6)
        return desc

    def forward(self, K, Rt, target=None):
        desc = self._build_desc(K, Rt, target)         # (B,V,16)
        logits = self.router(desc) / self.temperature   # (B,V,Kp)
        alpha = torch.softmax(logits, dim=-1)
        emb = alpha @ self.prototypes                   # (B,V,dim)
        return emb, {"alpha": alpha}

# EpiSoftCrossAttention

def _pixel_grid(H, W, device):
    u = torch.linspace(0, W-1, W, device=device)
    v = torch.linspace(0, H-1, H, device=device)
    vv, uu = torch.meshgrid(v, u, indexing="ij")
    return torch.stack([uu, vv, torch.ones_like(uu)], dim=-1).view(-1,3)  # (N,3)

def _fundamental(Ki, Rti, Kj, Rtj):
    Pi = Ki @ Rti; Pj = Kj @ Rtj                      # (3,4)
    Ri, ti = Rti[:,:3], Rti[:,3]
    Ci = -Ri.T @ ti
    e_p = Pj @ torch.cat([Ci, torch.ones(1, device=Ci.device)], dim=0)
    ex = e_p.new_tensor([[0, -e_p[2], e_p[1]],[e_p[2],0,-e_p[0]],[-e_p[1],e_p[0],0]])
    return ex @ Pj @ torch.linalg.pinv(Pi)            # (3,3)

class EpiSoftCrossAttention(nn.Module):
    def __init__(self, dim:int, num_heads:int=4, ds:int=2, lambda_epi:float=0.05, temperature:float=1.0):
        super().__init__()
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.heads = num_heads; self.ds = ds
        self.lambda_epi = lambda_epi; self.temperature = temperature

    def _pool(self, fmap):  # (B,V,D,H,W) -> (B,V,D,Hs,Ws)
        if self.ds <= 1: return fmap
        B,V,D,H,W = fmap.shape
        x = F.avg_pool2d(fmap.view(B*V, D, H, W), kernel_size=self.ds, stride=self.ds)
        return x.view(B,V,D,H//self.ds,W//self.ds)

    def forward(self, feat, K, Rt):
        """
        feat: (B,V,D,H,W),  K:(B,V,3,3) or (V,3,3),  Rt:(B,V,3,4) or (V,3,4)
        return: fused feat (B,V,D,H,W)
        """
        B,V,D,H,W = feat.shape
        small = self._pool(feat)                    # (B,V,D,Hs,Ws)
        Hs, Ws = small.shape[-2:]; N = Hs*Ws
        grid = _pixel_grid(Hs, Ws, feat.device)     # (N,3)

        tok = small.permute(0,1,3,4,2).reshape(B,V,N,D)   # (B,V,N,D)
        Q = self.q(tok); Kt = self.k(tok); Vt = self.v(tok)
        scale = (D // self.heads) ** -0.5 / self.temperature

        fused = torch.zeros_like(tok)
        for i in range(V):
            Zi_acc = []
            for j in range(V):
                # 기하 바이어스 계산 (배치 루프)
                bias = []
                for b in range(B):
                    Ki = K[b,i] if K.dim()==4 else K[i];   Rti = Rt[b,i] if Rt.dim()==4 else Rt[i]
                    Kj = K[b,j] if K.dim()==4 else K[j];   Rtj = Rt[b,j] if Rt.dim()==4 else Rt[j]
                    Fij = _fundamental(Ki, Rti, Kj, Rtj)          # (3,3)
                    l = (Fij @ grid.t())                           # (3,N)
                    a,bx,c = l[0].unsqueeze(1), l[1].unsqueeze(1), l[2].unsqueeze(1)
                    denom = torch.sqrt(a*a + bx*bx) + 1e-8
                    dist = (a*grid[:,0] + bx*grid[:,1] + c.squeeze(1)).abs()/denom.squeeze(1) # (N,)
                    bias.append(dist)  # (N,)
                bias = torch.stack(bias, dim=0)           # (B,N)

                logits = torch.einsum("bnd,bmd->bnm", Q[:,i], Kt[:,j]) * scale  # (B,N,N)
                logits = logits - self.lambda_epi * (bias.unsqueeze(1).expand(-1,N,-1) ** 2)
                A = torch.softmax(logits, dim=-1)
                Z = torch.einsum("bnm,bmd->bnd", A, Vt[:,j])  # (B,N,D)
                Zi_acc.append(Z)
            fused[:, i] = torch.stack(Zi_acc, dim=0).mean(dim=0)   # 모든 j 평균(원하면 i≠j만)

        fused_small = fused.view(B,V,Hs,Ws,D).permute(0,1,4,2,3).contiguous()  # (B,V,D,Hs,Ws)
        if self.ds > 1:
            up = F.interpolate(fused_small.view(B*V,D,Hs,Ws), size=(H,W), mode="bilinear", align_corners=False)
            up = up.view(B,V,D,H,W)
        else:
            up = fused_small
        return feat + self.o(up)  # residual


def build_2d_sincos_pos_embed(H: int, W: int, dim: int, device):
    """(H,W,dim) 2D sin-cos PE. dim은 짝수 추천."""
    assert dim % 4 == 0
    yy, xx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing="ij")
    omega = torch.arange(dim // 4, device=device) / (dim // 4)
    omega = 1. / (10000 ** omega)
    out = []
    for coord in (xx, yy):
        emb = torch.einsum("hw,c->hwc", coord, omega)
        out += [emb.sin(), emb.cos()]
    return torch.cat(out, dim=-1)  # (H,W,dim)

def views_to_tokens(feat):  # (B,V,D,Hf,Wf) -> (B, V*N, D), (Hf,Wf)
    B,V,D,Hf,Wf = feat.shape
    tok = feat.permute(0,1,3,4,2).reshape(B, V*Hf*Wf, D).contiguous()
    return tok, (Hf, Wf)

def tokens_to_views(tok, V: int, Hf:int, Wf:int):  # (B, V*N, D) -> (B,V,D,Hf,Wf)
    B, VN, D = tok.shape
    assert VN % (V*Hf*Wf) == 0
    return tok.view(B, V, Hf, Wf, D).permute(0,1,4,2,3).contiguous()

def best_hw_from_N(N: int):
    # N의 약수 중 H<=W, |H-W| 최소
    best = (int(math.sqrt(N)), N // int(math.sqrt(N)))
    min_gap = abs(best[0]-best[1])
    for h in range(1, int(math.sqrt(N))+1):
        if N % h == 0:
            w = N // h
            if w >= h and abs(w-h) < min_gap:
                best, min_gap = (h, w), abs(w-h)
    return best

def make_mem_pad_mask(valid_views: torch.Tensor, Hf:int, Wf:int):
    """
    valid_views: (B,V) 1=존재, 0=없음. -> mask (B, V*Hf*Wf), True=pad
    """
    B,V = valid_views.shape
    mask_v = (valid_views == 0)  # (B,V)
    mask_v = mask_v.unsqueeze(-1).expand(B, V, Hf*Wf).reshape(B, V*Hf*Wf)
    return mask_v

class DINOv3Backbone(nn.Module):
    """
    DINOv3 wrapper (+ optional projection to out_dim)
      images: (B,V,C,H,W) or (B,C,H,W)
      returns patch_tokens: (B,V,N,out_dim) / (B,N,out_dim)
              feature_map:  (B,V,out_dim,Hf,Wf) / (B,out_dim,Hf,Wf)
    """
    def __init__(self, model_name: str, freeze: bool = True, out_dim: Optional[int] = None):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.freeze = freeze

        # in/out dim은 ConvNeXt일 경우 config에 명시적이지 않을 수 있으므로 런타임에 확정
        self.in_dim  = None             # 첫 forward에서 결정
        self.out_dim = out_dim          # None이면 첫 forward에서 in_dim로 동기화
        self.proj_tok = None            # 첫 forward에서 필요시 생성
        self.proj_map = None

        if self.freeze:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

    def _flat_views(self, x: torch.Tensor):
        if x.dim()==5:
            B,V,C,H,W = x.shape
            return x.view(B*V, C, H, W), V
        elif x.dim()==4:
            return x, None
        raise ValueError("images must be 4D or 5D")

    def _unflat(self, t: torch.Tensor, V: Optional[int]):
        if V is None: return t
        Bv = t.shape[0]; B = Bv//V
        return t.view(B, V, *t.shape[1:])

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        import math

        x, V = self._flat_views(images)

        # 백본 호출만 no_grad (freeze일 때). 이후 프로젝션/reshape 등은 grad 허용.
        ctx = nullcontext() if not self.freeze else torch.no_grad()
        with ctx:
            out = self.model(pixel_values=x)
        hs = getattr(out, "last_hidden_state", None)
        if hs is None:
            raise RuntimeError("Backbone output has no last_hidden_state")

        # --- 채널(in_dim)과 아웃(out_dim) 확정 & 동적 프로젝션 준비 ---
        if hs.dim() == 3:
            # ViT 계열: (B*V, S, Din)
            Din = hs.size(-1)
        elif hs.dim() == 4:
            # ConvNeXt 계열: (B*V, C, Hf, Wf)
            Din = hs.size(1)
        else:
            raise RuntimeError(f"Unexpected backbone output shape: {tuple(hs.shape)}")

        if self.in_dim is None:
            self.in_dim = int(Din)
        if self.out_dim is None:
            self.out_dim = int(self.in_dim)

        need_proj = (self.out_dim != Din)
        if need_proj and (self.proj_tok is None or self.proj_map is None):
            # 런타임에 투영 레이어 생성 (학습 가능)
            self.proj_tok = nn.Linear(Din, self.out_dim, bias=False).to(hs.device)
            self.proj_map = nn.Conv2d(Din, self.out_dim, 1, bias=False).to(hs.device)

        # --- ViT 경로 ---
        if hs.dim() == 3:
            Bv, S, Din2 = hs.shape
            # DINOv3 ViT일 때만 CLS/등록토큰 제거
            num_register_tokens = int(getattr(getattr(self.model, "config", object()), "num_register_tokens", 0))
            has_cls = bool(getattr(getattr(self.model, "config", object()), "add_pooling_token", True)) or (num_register_tokens >= 0)
            start = 1 if has_cls and S > 0 else 0
            start += num_register_tokens
            patch_tokens = hs[:, start:, :]                               # (B*V, N, Din)
            N = patch_tokens.size(1)
            Hf = int(math.sqrt(N)); Wf = Hf if Hf*Hf==N else best_hw_from_N(N)[1]

            fmap = patch_tokens.permute(0,2,1).reshape(Bv, Din, Hf, Wf).contiguous()   # (B*V, Din, Hf, Wf)

        # --- ConvNeXt 경로 ---
        else:  # hs.dim() == 4
            Bv, Din2, Hf, Wf = hs.shape
            fmap = hs                                                        # (B*V, Din, Hf, Wf)
            patch_tokens = hs.permute(0,2,3,1).reshape(Bv, Hf*Wf, Din).contiguous()    # (B*V, N, Din)

        # --- 프로젝션 적용 (필요시) ---
        if need_proj and self.proj_tok is not None:
            patch_tokens = self.proj_tok(patch_tokens)                       # (B*V, N, out_dim)
        if need_proj and self.proj_map is not None:
            fmap = self.proj_map(fmap)                                       # (B*V, out_dim, Hf, Wf)

        return {
            "patch_tokens": self._unflat(patch_tokens, V),   # (B,V,N,out_dim) or (B,N,out_dim)
            "feature_map":  self._unflat(fmap, V),           # (B,V,out_dim,Hf,Wf) or (B,out_dim,Hf,Wf)
        }


# -------------------------
# 2) Per-view Kpt Head (baseline)
# -------------------------
class KptHead2D(nn.Module):
    def __init__(self, in_dim: int, num_joints: int, heatmap_size: Tuple[int,int]=(128,128)):
        super().__init__()
        self.heatmap_size = heatmap_size
        hid = max(128, in_dim//2)
        self.up = nn.Sequential(
            nn.Conv2d(in_dim, hid, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(hid, hid, 3, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(hid, hid, 2, stride=2), nn.ReLU(True),
            nn.ConvTranspose2d(hid, hid, 2, stride=2), nn.ReLU(True),
        )
        self.to_heat = nn.Conv2d(hid, num_joints, 1)

    def _head_single(self, feat4d):
        y = self.up(feat4d)          # (B,Hid,H',W')
        h = self.to_heat(y)          # (B,J,H',W')
        if self.heatmap_size is not None:
            h = F.interpolate(h, size=self.heatmap_size, mode="bilinear", align_corners=False)
        return h

    def forward(self, feat):
        if feat.dim()==5:  # (B,V,D,Hf,Wf)
            B,V,_,_,_ = feat.shape
            outs = [ self._head_single(feat[:,v]) for v in range(V) ]
            return torch.stack(outs, dim=1)  # (B,V,J,Hh,Wh)
        elif feat.dim()==4:  # (B,D,Hf,Wf)
            return self._head_single(feat)
        raise ValueError("feat must be (B,D,H,W) or (B,V,D,H,W)")

# -------------------------
# 3) Fusion Adapters (stubs)
# -------------------------
class EarlyFusionAdapter(nn.Module):
    """(B,V,D,Hf,Wf)->(B,D',Hf,Wf). 채널 concat 후 1x1 축소 옵션."""
    def __init__(self, in_dim:int, views:int, reduce_dim:Optional[int]=None):
        super().__init__()
        out_dim = in_dim * views
        self.reduce = nn.Conv2d(out_dim, reduce_dim, 1) if reduce_dim and reduce_dim<out_dim else None
        self.out_dim = reduce_dim if self.reduce is not None else out_dim

    def forward(self, feat):
        B,V,D,Hf,Wf = feat.shape
        x = feat.permute(0,1,3,4,2).reshape(B, V*D, Hf, Wf).contiguous()
        return self.reduce(x) if self.reduce is not None else x

class MiddleFusionAdapter(nn.Module):
    """
    자리만: 이후 epipolar/전범위 cross-attn 모듈을 꽂을 위치.
    현재는 identity(안정)로 동작.
    """
    def __init__(self, dim:int):
        super().__init__()
        self.stub = nn.Conv2d(dim, dim, 1)
    def forward(self, feat, intrinsics=None, extrinsics=None):
        return feat + 0.0*self.stub(feat)


# -------------------------
# 4) 전체 토글 가능한 모델
# -------------------------
class MultiViewPoseNet(nn.Module):
    def __init__(self, num_joints:int,
                 fusion:Literal["auto","late","middle","early"]="auto",
                 default_fusion_for_multi:Literal["late","middle","early"]="late",
                 backbone_name:str=MODEL_CNX,
                 freeze_backbone:bool=True,
                 max_views:int=8,
                 early_reduce_dim:Optional[int]=None,
                 middle_impl:Optional[nn.Module]=None,
                 feature_dim: Optional[int] = None,
                 heatmap_size: Tuple[int,int] = (128,128)):
        super().__init__()
        self.fusion = fusion
        self.default_fusion_for_multi = default_fusion_for_multi
        self.max_views = max_views

        # Backbone with out_dim projection
        self.backbone = DINOv3Backbone(backbone_name, freeze=freeze_backbone, out_dim=feature_dim)
        D = self.backbone.out_dim

        # Heads / Adapters
        self.head_per_view = KptHead2D(D, num_joints, heatmap_size=heatmap_size)
        self.early_adapter = EarlyFusionAdapter(D, max_views, early_reduce_dim) if default_fusion_for_multi=="early" or fusion=="early" else None
        self.head_early    = KptHead2D(self.early_adapter.out_dim if self.early_adapter else D, num_joints, heatmap_size=heatmap_size) if (self.early_adapter or fusion=="early") else None

        # Middle 구현(없으면 identity)
        self.middle = middle_impl if middle_impl is not None else MiddleFusionAdapter(D)


    def _run_single(self, fmap):  # fmap: (B,1,D,Hf,Wf)
        # 싱글뷰는 per-view head 한 장만 처리
        heat = self.head_per_view(fmap)  # (B,1,J,Hh,Wh)
        return heat

    def _run_multi(self, fmap, intrinsics=None, extrinsics=None):
        # fmap: (B,V,D,Hf,Wf), V>1
        mode = self.default_fusion_for_multi if self.fusion in ("auto",) else self.fusion

        if mode == "early":
            V = fmap.size(1)
            if V < self.max_views:
                B, _, D, Hf, Wf = fmap.shape
                pad = torch.zeros(B, self.max_views-V, D, Hf, Wf, device=fmap.device, dtype=fmap.dtype)
                fmap = torch.cat([fmap, pad], dim=1)
            fused = self.early_adapter(fmap)       # (B,D',Hf,Wf)
            heat  = self.head_early(fused).unsqueeze(1)  # (B,1,J,Hh,Wh) ← early는 단일 맵
        elif mode == "middle":
            fused = self.middle(fmap, intrinsics, extrinsics)  # (B,V,D,Hf,Wf)
            heat  = self.head_per_view(fused)                  # (B,V,J,Hh,Wh)
        else:  # "late"
            heat  = self.head_per_view(fmap)                   # (B,V,J,Hh,Wh)
        return heat

    def forward(self, images, intrinsics=None, extrinsics=None, valid_views:Optional[torch.Tensor]=None):
        """
        images: (B,C,H,W) or (B,1,C,H,W) or (B,V,C,H,W)
        returns:
          {"heatmaps": (B,V',J,Hh,Wh), "Hf":Hf, "Wf":Wf}
        """
        bk   = self.backbone(images)
        fmap = bk["feature_map"]                           # (B,V,D,Hf,Wf) or (B,D,Hf,Wf)
        if fmap.dim() == 4: fmap = fmap.unsqueeze(1)       # (B,1,D,Hf,Wf)
        B, V, D, Hf, Wf = fmap.shape

        # fusion 경로 결정
        if self.fusion == "auto":
            if V == 1:
                heat = self._run_single(fmap)              # (B,1,J,Hh,Wh)
            else:
                heat = self._run_multi(fmap, intrinsics, extrinsics)
        elif self.fusion in ("late","middle","early"):
            if V == 1 and self.fusion != "early":
                heat = self._run_single(fmap)
            else:
                heat = self._run_multi(fmap, intrinsics, extrinsics)
        else:
            raise ValueError(f"Unknown fusion mode: {self.fusion}")

        return {"heatmaps": heat, "Hf": Hf, "Wf": Wf}

class TokenLevelFusion(nn.Module):
    """
    Patch-token 레벨 글로벌 컨텍스트 추출 + 토큰 재주입
    - 입력: patch_tokens (B,V,N,D) 또는 (B,N,D)
    - 옵션: 카메라가 있으면 AutoViewEmbed 기반의 view embedding 사용, 없으면 learnable view embedding
    - 출력:
        fmap_tok: (B,V,D,Hf,Wf)  # 토큰 보강 후 맵
        latent:   (B,Q,D)        # 글로벌 잠재 (3D 헤드 입력용)
    """
    def __init__(self, dim:int, max_views:int=8, num_queries:int=16, num_heads:int=8, num_layers:int=2,
                 use_auto_view: bool = True):
        super().__init__()
        self.dim = dim
        self.max_views = max_views
        self.use_auto_view = use_auto_view

        # 글로벌 컨텍스트 추출기(기존 디코더 재사용)
        self.fuser = MultiViewFusion(dim, num_heads=num_heads, num_queries=num_queries, num_layers=num_layers)
        self.inject = LatentToViews(dim)

        # view embedding (fallback)
        self.view_embed = nn.Embedding(max_views, dim)

        # 카메라 기반 뷰 임베딩(선택)
        if use_auto_view:
            self.auto_view = AutoViewEmbed(dim, num_prototypes=8, temperature=1.0)
        else:
            self.auto_view = None

    def _ensure_BVN(self, patch_tokens: torch.Tensor):
        # (B,V,N,D) or (B,N,D) -> (B,V,N,D)
        if patch_tokens.dim() == 3:
            B,N,D = patch_tokens.shape
            return patch_tokens.unsqueeze(1), 1, N, D
        elif patch_tokens.dim() == 4:
            B,V,N,D = patch_tokens.shape
            return patch_tokens, V, N, D
        else:
            raise ValueError("patch_tokens must be (B,N,D) or (B,V,N,D)")

    def forward(self, patch_tokens: torch.Tensor,
                Ks: Optional[torch.Tensor]=None,
                Rts: Optional[torch.Tensor]=None,
                valid_views: Optional[torch.Tensor]=None):
        tok, V, N, D = self._ensure_BVN(patch_tokens)     # (B,V,N,D)
        B = tok.size(0)

        # Hf,Wf 복원
        Hf = int(math.sqrt(N))
        if Hf*Hf != N: Hf, Wf = best_hw_from_N(N)
        else:          Wf = Hf

        # 2D sin-cos PE
        pe2d = build_2d_sincos_pos_embed(Hf, Wf, D, tok.device).view(1,1,Hf,Wf,D)
        pe2d = pe2d.expand(B, V, -1, -1, -1).reshape(B, V, N, D)     # (B,V,N,D)

        # View embedding (카메라 기반 우선)
        if self.auto_view is not None and (Ks is not None) and (Rts is not None):
            vemb, _ = self.auto_view(Ks, Rts, target=None)           # (B,V,D)
        else:
            ids = torch.arange(V, device=tok.device).view(1,V).expand(B,V)
            vemb = self.view_embed(ids)                              # (B,V,D)
        vemb = vemb.unsqueeze(2).expand(B, V, N, D)                  # (B,V,N,D)

        # 메모리 구성
        mem = tok + pe2d + vemb                                      # (B,V,N,D)
        mem_flat = mem.reshape(B, V*N, D)                             # (B,VN,D)

        # (신규) pad mask: 없어진/채워진 뷰 토큰을 무시
        mem_pad_mask = None
        if valid_views is not None:
            # valid_views: (B,V) 1/0
            mem_pad_mask = make_mem_pad_mask(valid_views.long(), Hf, Wf)  # (B, V*Hf*Wf)

        # 글로벌 latent
        latent = self.fuser(tokens=mem_flat, pos_mem=None, mem_pad_mask=mem_pad_mask)  # (B,Q,D)

        # latent → per-token 재주입
        fused_flat = self.inject(latent, mem_flat, view_mask=mem_pad_mask)             # (B,VN,D)
        fused = fused_flat.view(B, V, N, D)

        # 토큰 → 맵
        fmap_tok = fused.view(B, V, Hf, Wf, D).permute(0,1,4,2,3).contiguous()         # (B,V,D,Hf,Wf)
        return fmap_tok, latent, (Hf, Wf)

    
class MultiViewFusion(nn.Module):
    def __init__(self, dim, num_heads=8, num_queries=16, num_layers=2, dropout=0.1):
        super().__init__()
        self.q_latent = nn.Parameter(torch.randn(1, num_queries, dim))  # global or joint queries
        layer = nn.TransformerDecoderLayer(d_model=dim, nhead=num_heads,
                                           dim_feedforward=dim*4, dropout=dropout,
                                           activation='gelu', batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        # 간단한 per-view 주입 레이어(선택)
        self.inject = nn.Linear(dim, dim)

    def forward(self, tokens, pos_mem=None, pos_q=None, mem_pad_mask=None):
        """
        tokens:     (B, V*N, D)   # concat된 memory
        pos_mem:    (B, V*N, D)   # 2D + view + camera pose pe
        pos_q:      (B, Q,   D)   # 쿼리용 pe (joint별이면 관절ID 임베딩 등)
        mem_pad_mask: (B, V*N)    # True=pad(무시)

        return:
          latent: (B, Q, D)
        """
        B = tokens.size(0)
        q = self.q_latent.expand(B, -1, -1)
        if pos_q is not None:
            q = q + pos_q

        memory = tokens if pos_mem is None else tokens + pos_mem

        latent = self.decoder(tgt=q, memory=memory,
                              memory_key_padding_mask=mem_pad_mask)
        return latent

# MiddleFusionWithLatent 교체: view 임베딩을 학습 파라미터로
class MiddleFusionWithLatent(nn.Module):
    def __init__(self, dim:int, num_heads=8, num_queries=None, num_layers=2, dropout=0.1, max_views: int = 8):
        super().__init__()
        self.dim = dim
        self.q = (num_queries or 16)
        self.fuser = MultiViewFusion(dim, num_heads, self.q, num_layers, dropout)
        self.inject = LatentToViews(dim)
        self.view_embed = nn.Embedding(max_views, dim)  # ← learnable, 고정 분포

    def forward(self, feat, intrinsics=None, extrinsics=None):
        tok, (Hf,Wf) = views_to_tokens(feat)   # (B,V*N,D)
        B, VN, D = tok.shape
        device = tok.device
        V = feat.shape[1]

        pos_2d = build_2d_sincos_pos_embed(Hf, Wf, D, device).view(1,1,Hf,Wf,D).expand(B,V,-1,-1,-1)
        pos_2d = pos_2d.reshape(B, V*Hf*Wf, D)

        view_ids = torch.arange(V, device=device).view(1,V).expand(B,V)  # (B,V)
        view_emb = self.view_embed(view_ids).unsqueeze(2).expand(B, V, Hf*Wf, D).reshape(B, V*Hf*Wf, D)

        pos_mem = pos_2d + view_emb
        latent = self.fuser(tok, pos_mem=pos_mem)     # (B,Q,D)
        fused_tok = self.inject(latent, tok)          # (B,V*N,D)
        fused = tokens_to_views(fused_tok, V, Hf, Wf) # (B,V,D,Hf,Wf)
        return fused


class LatentToViews(nn.Module):
    """ 글로벌 latent를 다시 각 뷰 feature로 주입(가벼운 cross-attn 흉내) """
    def __init__(self, dim):
        super().__init__()
        self.proj_q = nn.Linear(dim, dim)
        self.proj_k = nn.Linear(dim, dim)
        self.proj_v = nn.Linear(dim, dim)
        self.proj_o = nn.Linear(dim, dim)

    def forward(self, latent, view_tokens, view_mask=None):
        """
        latent:      (B, Q, D)
        view_tokens: (B, V*N, D)
        view_mask:   (B, V*N)  True=pad
        returns:
          fused_tokens: (B, V*N, D)
        """
        Q = self.proj_q(view_tokens)                 # queries: per-token
        K = self.proj_k(latent)                      # keys: latent
        V = self.proj_v(latent)
        attn_logits = (Q @ K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)  # (B, V*N, Q)

        if view_mask is not None:
            attn_logits = attn_logits.masked_fill(view_mask.unsqueeze(-1), float('-inf'))

        A = torch.softmax(attn_logits, dim=-1)       # (B, V*N, Q)
        Z = A @ V                                    # (B, V*N, D)
        fused = view_tokens + self.proj_o(Z)         # residual
        return fused
    
    
class MiddleFusionEpiSoft(nn.Module):
    def __init__(self, dim:int, num_heads:int=4, ds:int=2, lambda_epi:float=0.05, temperature:float=1.0,
                 num_view_prototypes:int=8):
        super().__init__()
        self.cam_pe = CameraPE(dim)
        self.view_auto = AutoViewEmbed(dim, num_prototypes=num_view_prototypes, temperature=1.0)
        self.epi = EpiSoftCrossAttention(dim, num_heads=num_heads, ds=ds,
                                         lambda_epi=lambda_epi, temperature=temperature)

    def forward(self, feat, Ks=None, Rts=None, target=None):
        """
        feat: (B,V,D,Hf,Wf)
        Ks:   (B,V,3,3) or (V,3,3) or None
        Rts:  (B,V,3,4) or (V,3,4) or None
        """
        B,V,D,H,W = feat.shape
        if Ks is None or Rts is None:
            # 카메라 정보가 없으면 그냥 패스(안전)
            return feat

        # 카메라/뷰 임베딩을 feat에 더해 안정화
        cam = self.cam_pe(Ks, Rts)                       # (B,V,D)
        vemb, _ = self.view_auto(Ks, Rts, target=target) # (B,V,D)
        cam = cam.view(B,V,D,1,1).expand(-1,-1,-1,H,W)
        vemb= vemb.view(B,V,D,1,1).expand(-1,-1,-1,H,W)
        feat = feat + cam + vemb

        # 저해상 cross-attn + 소프트 에피폴라 바이어스
        return self.epi(feat, Ks, Rts)


class SoftArgmax2D(nn.Module):
    def __init__(self, normalize: bool = True, eps: float = 1e-9):
        super().__init__()
        self.normalize = normalize
        self.eps = eps

    def forward(self, heat: torch.Tensor):
        """
        heat: (B, J, H, W) or (B, V, J, H, W)
        return:
          coords: (B, J, 2) or (B, V, J, 2)  # (u, v)
          conf:   (B, J)   or (B, V, J)
        """
        if heat.dim() == 5:
            B, V, J, H, W = heat.shape
            x = heat.view(B*V, J, H, W)
            coords, conf = self._forward_single(x)
            coords = coords.view(B, V, J, 2)
            conf = conf.view(B, V, J)
            return coords, conf
        elif heat.dim() == 4:
            return self._forward_single(heat)
        else:
            raise ValueError

    def _forward_single(self, h):
        B, J, H, W = h.shape
        h = h.view(B, J, -1)
        h = torch.softmax(h, dim=-1) + self.eps
        h = h / h.sum(dim=-1, keepdim=True)

        # coordinate grids
        u = torch.linspace(0, W-1, W, device=h.device)
        v = torch.linspace(0, H-1, H, device=h.device)
        uu, vv = torch.meshgrid(v, u, indexing="ij") # (H,W) note (v,u) order
        uu = uu.reshape(-1); vv = vv.reshape(-1)

        u_exp = (h * vv)  # (B,J,HW)
        v_exp = (h * uu)
        u_hat = u_exp.sum(dim=-1)
        v_hat = v_exp.sum(dim=-1)
        conf = h.max(dim=-1).values  # peak confidence (soft)

        coords = torch.stack([u_hat, v_hat], dim=-1)  # (B,J,2)
        return coords, conf

class DINOv3PoseEstimator(nn.Module):
    def __init__(self, cfg: Optional[ModelCfg] = None, **kwargs):
        """
        선호: DINOv3PoseEstimator(cfg=ModelCfg(...))
        호환: DINOv3PoseEstimator(model_id="facebook/...") 또는 model_name=..., num_joints=..., num_angles=...
        """
        if cfg is None:
            # 과거 스타일 허용
            model_name = kwargs.pop("model_id", kwargs.pop("model_name", MODEL_CNX))
            num_angles = int(kwargs.pop("num_angles", 6))
            num_joints = int(kwargs.pop("num_joints", 8))
            # 나머지 선택 인자들 (없으면 ModelCfg 기본 사용)
            cfg = ModelCfg(
                MODEL_NAME=model_name,
                NUM_ANGLES=num_angles,
                NUM_JOINTS=num_joints,
                FEATURE_DIM=kwargs.pop("feature_dim", 768),
                HEATMAP_SIZE=kwargs.pop("heatmap_size", (128,128)),
                MAX_VIEWS_PER_GROUP=kwargs.pop("max_views", 8),
                FUSION=kwargs.pop("fusion", "auto"),
                DEFAULT_FUSION_FOR_MULTI=kwargs.pop("default_fusion_for_multi", "late"),
                FREEZE_BACKBONE=kwargs.pop("freeze_backbone", True),
                MIDDLE_HEADS=kwargs.pop("middle_heads", 4),
                MIDDLE_DS=kwargs.pop("middle_ds", 2),
                MIDDLE_LAMBDA_EPI=kwargs.pop("middle_lambda_epi", 0.05),
                MIDDLE_TEMPERATURE=kwargs.pop("middle_temperature", 1.0),
                MIDDLE_NUM_VIEW_PROTOTYPES=kwargs.pop("middle_num_view_prototypes", 8),
                EARLY_REDUCE_DIM=kwargs.pop("early_reduce_dim", None),
                TOKEN_NUM_QUERIES=kwargs.pop("token_num_queries", 16),
                TOKEN_NUM_HEADS=kwargs.pop("token_num_heads", 8),
                TOKEN_NUM_LAYERS=kwargs.pop("token_num_layers", 2),
                USE_AUTO_VIEW_FOR_TOKENS=kwargs.pop("use_auto_view_for_tokens", True),
            )
        super().__init__()
        self.cfg = cfg

        # 1) 기존 네트(백본/헤드/에피폴라) - cfg 반영
        self.net = MultiViewPoseNet(
            num_joints=cfg.NUM_JOINTS,
            fusion=cfg.FUSION,
            default_fusion_for_multi=cfg.DEFAULT_FUSION_FOR_MULTI,
            backbone_name=cfg.MODEL_NAME,
            freeze_backbone=cfg.FREEZE_BACKBONE,
            max_views=cfg.MAX_VIEWS_PER_GROUP,
            early_reduce_dim=cfg.EARLY_REDUCE_DIM,
            middle_impl=MiddleFusionAdapter(1),     # placeholder
            feature_dim=cfg.FEATURE_DIM,
            heatmap_size=cfg.HEATMAP_SIZE,
        )
        D = self.net.backbone.out_dim
        self.net.middle = MiddleFusionEpiSoft(
            dim=D,
            num_heads=cfg.MIDDLE_HEADS,
            ds=cfg.MIDDLE_DS,
            lambda_epi=cfg.MIDDLE_LAMBDA_EPI,
            temperature=cfg.MIDDLE_TEMPERATURE,
            num_view_prototypes=cfg.MIDDLE_NUM_VIEW_PROTOTYPES
        )

        # 2) 토큰 레벨 글로벌 컨텍스트
        self.token_fusion = TokenLevelFusion(
            dim=D, max_views=cfg.MAX_VIEWS_PER_GROUP,
            num_queries=cfg.TOKEN_NUM_QUERIES,
            num_heads=cfg.TOKEN_NUM_HEADS,
            num_layers=cfg.TOKEN_NUM_LAYERS,
            use_auto_view=cfg.USE_AUTO_VIEW_FOR_TOKENS
        )

        # 3) 2D 좌표 추정기 & 4) 3D 헤드 & 5) 로스 유틸
        self.decoder_2d = SoftArgmax2D()
        self.head_3d = Pose3DHead(dim=D, num_joints=cfg.NUM_JOINTS, num_queries=cfg.TOKEN_NUM_QUERIES)
        self.triang_loss = TriangulationConsistencyLoss(reduction="mean")


    # 내부: 백본 실행만
    def _run_backbone(self, images: torch.Tensor):
        bk = self.net.backbone(images)  # {"patch_tokens":(B,V,N,D)/(B,N,D), "feature_map":(B,V,D,H,W)/(B,D,H,W)}
        patch = bk["patch_tokens"]
        fmap  = bk["feature_map"]
        if patch.dim()==3: patch = patch.unsqueeze(1)
        if fmap.dim()==4:  fmap  = fmap.unsqueeze(1)
        return patch, fmap  # (B,V,N,D), (B,V,D,H,W)
    
    def _stack_by_keys(data_dict, keys):
        """ {k: Tensor(B, ...)} -> Tensor(B,V,...) (키 순서 기준) """
        ts = [data_dict[k] for k in keys]
        return torch.stack(ts, dim=1)

    @staticmethod
    def _images_to_batched(images):
        """
        images:
          - Tensor: (B,V,C,H,W) or (B,C,H,W)
          - dict:
             {key: image(B,C,H,W)} 또는
             {key: {"image": x, "K": K, "Rt": Rt}}
        return: images_batched(B,V,C,H,W), Ks(Optional), Rts(Optional), keys(list|None)
        """
        Ks = Rts = None
        keys = None
        if isinstance(images, dict):
            keys = list(images.keys())
            # 값이 Tensor 인 경우
            if torch.is_tensor(next(iter(images.values()))):
                imgs = {k: images[k] for k in keys}
                x = DINOv3PoseEstimator._stack_by_keys(imgs, keys)  # (B,V,C,H,W)
            else:
                # 값이 dict 인 경우
                imgs = {k: images[k]["image"] for k in keys}
                x = DINOv3PoseEstimator._stack_by_keys(imgs, keys)
                if all(("K" in images[k] and images[k]["K"] is not None) for k in keys):
                    kdict = {k: images[k]["K"] for k in keys}
                    Ks = DINOv3PoseEstimator._stack_by_keys(kdict, keys)  # (B,V,3,3)
                if all(("Rt" in images[k] and images[k]["Rt"] is not None) for k in keys):
                    rdict = {k: images[k]["Rt"] for k in keys}
                    Rts = DINOv3PoseEstimator._stack_by_keys(rdict, keys) # (B,V,3,4)
        else:
            x = images
            if x.dim() == 4:  # (B,C,H,W) → (B,1,C,H,W)
                x = x.unsqueeze(1)
        return x, Ks, Rts, keys

    @staticmethod
    def _split_to_dict(keys, tensor_per_view):
        """ (B,V,...) -> dict(view_key -> (B,...)) """
        B, V = tensor_per_view.shape[:2]
        return {keys[i]: tensor_per_view[:, i].contiguous() for i in range(V)}

    # 편의: 런타임에서 fusion 모드 교체
    def set_fusion_mode(self, fusion: Literal["auto","late","middle","early"],
                        default_fusion_for_multi: Optional[Literal["late","middle","early"]] = None):
        self.net.fusion = fusion
        if default_fusion_for_multi is not None:
            self.net.default_fusion_for_multi = default_fusion_for_multi
            
    def forward(self,
                images: Union[torch.Tensor, dict],
                Ks: Optional[Union[torch.Tensor, dict]] = None,
                Rts: Optional[Union[torch.Tensor, dict]] = None,
                valid_views: Optional[torch.Tensor] = None,
                target_point: Optional[torch.Tensor] = None,
                as_dict: bool = False,
                return_3d: bool = True,
                gt_2d: Optional[torch.Tensor] = None,  # (B,V,J,2) normalized or pixel
                lambda_triang: float = 0.1,
                heatmap_size: Optional[Tuple[int,int]] = None):
        """
        - images: Tensor (B,V,C,H,W)/(B,C,H,W) or dict (키별)
        - Ks,Rts: (B,V,3,3)/(B,V,3,4) or dict
        - gt_2d:  (선택) L_triang 계산시 사용. 픽셀 좌표 기준으로 맞춰주면 좋음.
        - heatmap_size: SoftArgmax2D로 얻은 좌표가 픽셀 스케일로 변환 필요할 때 지정(예: (W,H)).
        """
        # 0) 입력 통일
        x, Ks_local, Rts_local, keys = DINOv3PoseEstimator._images_to_batched(images)

        # Ks/Rts override (dict 허용)
        if Ks is not None:
            if isinstance(Ks, dict):
                Ks_local, _, _, _ = DINOv3PoseEstimator._images_to_batched(Ks)
            else:
                Ks_local = Ks
        if Rts is not None:
            if isinstance(Rts, dict):
                Rts_local, _, _, _ = DINOv3PoseEstimator._images_to_batched(Rts)
            else:
                Rts_local = Rts

        # 1) 백본
        patch_tokens, fmap0 = self._run_backbone(x)                   # (B,V,N,D), (B,V,D,H,W)

        # 2) 토큰 레벨 글로벌 컨텍스트
        fmap_tok, latent, (Hf, Wf) = self.token_fusion(
            patch_tokens, Ks_local, Rts_local, valid_views=valid_views
        )

        # 3) (옵션) middle 에피폴라 보강/또는 late/early 경로
        V = fmap_tok.size(1)
        mode = self.net.fusion
        if mode == "auto":
            mode = self.net.default_fusion_for_multi if V > 1 else "late"

        if mode == "early":
            # early는 채널 concat 후 단일 heatmap
            if V < self.net.max_views:
                B, _, D, H, W = fmap_tok.shape
                pad = torch.zeros(B, self.net.max_views-V, D, H, W, device=fmap_tok.device, dtype=fmap_tok.dtype)
                fmap_for_head = torch.cat([fmap_tok, pad], dim=1)
            else:
                fmap_for_head = fmap_tok
            fused_early = self.net.early_adapter(fmap_for_head)       # (B,D',H,W)
            heat = self.net.head_early(fused_early).unsqueeze(1)      # (B,1,J,Hh,Wh)
        elif mode == "middle":
            fmap_mid = self.net.middle(fmap_tok, Ks_local, Rts_local, target=target_point)  # (B,V,D,H,W)
            heat = self.net.head_per_view(fmap_mid)                   # (B,V,J,Hh,Wh)
        else:  # "late"
            heat = self.net.head_per_view(fmap_tok)                   # (B,V,J,Hh,Wh)

        # 4) 2D 좌표/신뢰도
        coords_2d, conf = self.decoder_2d(heat)                       # (B,V,J,2), (B,V,J)

        # 히트맵 해상도 → 픽셀 좌표로 스케일 맞추기(선택)
        if heatmap_size is not None:
            Wp, Hp = heatmap_size
            # SoftArgmax는 [0..W-1],[0..H-1] 범위였으므로 그대로 픽셀 좌표
            # 필요 시 추가 변환 로직 삽입 가능
            pass

        # 5) 3D 추정 및 재투영(선택)
        out = {"heatmaps": heat, "coords": coords_2d, "conf": conf, "Hf": Hf, "Wf": Wf}
        if keys is not None:
            out["view_keys"] = keys

        if return_3d:
            P3d = self.head_3d(latent)                                # (B,J,3)
            out["coords_3d"] = P3d
            if (Ks_local is not None) and (Rts_local is not None):
                reproj = project_points(P3d, Ks_local, Rts_local)     # (B,V,J,2)
                out["coords_reproj"] = reproj

                # (선택) 학습 중 일관성 로스 계산
                if (self.training and (gt_2d is not None)):
                    # gt_2d가 픽셀 좌표로 들어왔다고 가정(필요 시 정규화/역정규화 맞추기)
                    loss_tr = self.triang_loss(reproj, gt_2d, conf)
                    out["loss_triangulation"] = loss_tr
                    out["loss_aux"] = lambda_triang * loss_tr

        # dict 반환 옵션
        if keys is not None and as_dict:
            out["heatmaps"] = DINOv3PoseEstimator._split_to_dict(keys, out["heatmaps"])
            out["coords"]   = DINOv3PoseEstimator._split_to_dict(keys, out["coords"])
            out["conf"]     = DINOv3PoseEstimator._split_to_dict(keys, out["conf"])
            if "coords_reproj" in out:
                out["coords_reproj"] = DINOv3PoseEstimator._split_to_dict(keys, out["coords_reproj"])

        return out
