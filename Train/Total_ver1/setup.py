# setup.py
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchvision.transforms as transforms

from dataset import UnifiedRobotPoseDataset
from loss_and_metrics import make_angle_loss
from models import DINOv3PoseEstimator, ModelCfg, MODEL_CNX  # MODEL_CNX는 기본 백본용 상수

# ---------------------------------------------------------
# dict-of-views 배치 패딩 collate
# ---------------------------------------------------------
def make_collate_pad_dicts():
    def collate_fn(batch):
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None, None

        image_dicts, heatmap_dicts, angles_list = zip(*batch)
        all_keys = sorted(list(set(itertools.chain.from_iterable(d.keys() for d in image_dicts))))

        sample_img_tensor  = next(iter(image_dicts[0].values()))
        sample_hmap_tensor = next(iter(heatmap_dicts[0].values()))
        dummy_img  = torch.zeros_like(sample_img_tensor)
        dummy_hmap = torch.zeros_like(sample_hmap_tensor)

        padded_images, padded_heatmaps = [], []
        for i in range(len(batch)):
            padded_img_dict  = {k: image_dicts[i].get(k,  dummy_img)  for k in all_keys}
            padded_hmap_dict = {k: heatmap_dicts[i].get(k, dummy_hmap) for k in all_keys}
            padded_images.append(padded_img_dict)
            padded_heatmaps.append(padded_hmap_dict)

        images_collated   = torch.utils.data.dataloader.default_collate(padded_images)
        heatmaps_collated = torch.utils.data.dataloader.default_collate(padded_heatmaps)
        angles_collated   = torch.stack(angles_list)
        return images_collated, heatmaps_collated, angles_collated
    return collate_fn


# ---------------------------------------------------------
# Experiment Setup
# ---------------------------------------------------------
def setup(dataset_type,            # 'fr3' | 'fr5' | 'meca500'
          dataset_items,           # groups/pairs
          hyperparameters,
          rank, world_size,
          model_cls,               # e.g. DINOv3PoseEstimator
          extra_model_kwargs=None  # ✅ 기본값 None로 변경 (hparams 참조 금지)
          ):
    print(f"--- [Rank {rank}] Setting up environment for {dataset_type.upper()} ---")
    device = torch.device(f"cuda:{rank}")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    resize_size  = hyperparameters.get("input_size", 224)
    heatmap_size = hyperparameters.get("heatmap_size", (128,128))
    sigma        = hyperparameters.get("sigma", 5.0)

    # --------- transforms ----------
    def build_base_transform(mean, std, resize_size=224):
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    def build_strong_transform(mean, std, resize_size=224):
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.15, hue=0.05),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    base_transform   = build_base_transform(mean, std, resize_size)
    strong_transform = build_strong_transform(mean, std, resize_size)

    # --------- split ----------
    torch.manual_seed(42 + rank)
    idx = torch.randperm(len(dataset_items)).tolist()
    n_train     = int(len(dataset_items) * (1 - hyperparameters['val_split']))
    train_items = [dataset_items[i] for i in idx[:n_train]]
    val_items   = [dataset_items[i] for i in idx[n_train:]]

    def _filter(items):
        out = []
        for it in items:
            if 'views' in it:
                if len(it['views']) >= 2:
                    out.append(it)
            else:
                out.append(it)
        return out
    train_items = _filter(train_items)
    val_items   = _filter(val_items)

    # --------- dataset ----------
    train_dataset = UnifiedRobotPoseDataset(
        dataset_type=dataset_type,
        items=train_items,
        transform=base_transform,
        heatmap_size=heatmap_size,
        sigma=sigma,
        input_size=resize_size,
        robot=dataset_type,
        robot_fk_unit=None,
    )
    val_dataset = UnifiedRobotPoseDataset(
        dataset_type=dataset_type,
        items=val_items,
        transform=base_transform,
        heatmap_size=heatmap_size,
        sigma=sigma,
        input_size=resize_size,
        robot=dataset_type,
        robot_fk_unit=None,
    )

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

    collate_fn = make_collate_pad_dicts()
    train_loader = DataLoader(
        train_dataset,
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters.get('num_workers', 8),
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hyperparameters['batch_size'],
        num_workers=hyperparameters.get('num_workers', 8),
        collate_fn=collate_fn,
        pin_memory=True,
        sampler=val_sampler
    )

    # --------- NUM_ANGLES ----------
    default_angles = {'fr3': 7, 'fr5': 6, 'meca500': 6}
    num_angles = default_angles.get(dataset_type, None)
    if num_angles is None:
        probe = None
        for k in range(min(8, len(train_dataset))):
            s = train_dataset[k]
            if s[0] is not None:
                probe = s; break
        if probe is not None:
            num_angles = int(probe[2].numel())
        else:
            raise RuntimeError("Could not infer NUM_ANGLES from dataset.")
    print(f"[Rank {rank}] NUM_ANGLES = {num_angles}")

    # --------- ModelCfg (extra_model_kwargs로 override 지원) ----------
    extra = extra_model_kwargs or {}
    backbone_name  = extra.get('model_name', MODEL_CNX)
    feature_dim    = extra.get('feature_dim', hyperparameters.get('feature_dim', 768))
    fusion_mode    = extra.get('fusion',        "auto")
    default_fusion = extra.get('default_fusion_for_multi', "late")
    freeze_bb      = extra.get('freeze_backbone', True)
    early_reduce   = extra.get('early_reduce_dim', None)

    cfg = ModelCfg(
        MODEL_NAME=backbone_name,
        NUM_ANGLES=num_angles,
        NUM_JOINTS=num_angles+1,
        FEATURE_DIM=feature_dim,
        HEATMAP_SIZE=tuple(heatmap_size),
        MAX_VIEWS_PER_GROUP=8,
        FUSION=fusion_mode,
        DEFAULT_FUSION_FOR_MULTI=default_fusion,
        FREEZE_BACKBONE=freeze_bb,
        MIDDLE_HEADS=4,
        MIDDLE_DS=2,
        MIDDLE_LAMBDA_EPI=0.05,
        MIDDLE_TEMPERATURE=1.0,
        MIDDLE_NUM_VIEW_PROTOTYPES=8,
        EARLY_REDUCE_DIM=early_reduce,
        TOKEN_NUM_QUERIES=16,
        TOKEN_NUM_HEADS=8,
        TOKEN_NUM_LAYERS=2,
        USE_AUTO_VIEW_FOR_TOKENS=True,
    )
    print(f"[Rank {rank}] Using backbone: {cfg.MODEL_NAME}")

    model = DINOv3PoseEstimator(cfg=cfg).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # --------- losses / criteria ----------
    angle_loss = make_angle_loss(num_angles, vm_weight=0.5, cos_weight=0.5)
    if hasattr(angle_loss, "vm"):
        angle_loss.vm = angle_loss.vm.to(device)

    fk_reg = None
    try:
        from loss_and_metrics import FKRegularizer
        fk_reg = FKRegularizer(robot=dataset_type, device=device)
    except Exception:
        fk_reg = None

    criteria = {'ang': angle_loss}
    criteria['fk'] = fk_reg if fk_reg is not None else None
    criteria['lambda_fk'] = hyperparameters.get('lambda_fk', 0.5 if fk_reg is not None else 0.0)

    # --------- Optimizers & param_sets ----------
    m = model.module if hasattr(model, "module") else model

    def _params(mod):
        if mod is None: return []
        return [p for p in mod.parameters() if p.requires_grad]

    kpt_modules = [
        getattr(m.net, "head_per_view", None),
        getattr(m.net, "head_early", None),
        getattr(m.net, "middle", None),
        getattr(m.net, "early_adapter", None),
    ]
    ang_modules = [
        getattr(m, "token_fusion", None),
        getattr(m, "head_3d", None),
    ]

    backbone_trainable = []
    if not cfg.FREEZE_BACKBONE:
        if getattr(m.net.backbone, "proj_tok", None) is not None:
            backbone_trainable += _params(m.net.backbone.proj_tok)
        if getattr(m.net.backbone, "proj_map", None) is not None:
            backbone_trainable += _params(m.net.backbone.proj_map)

    params_kpt = []
    for mod in kpt_modules: params_kpt += _params(mod)
    params_kpt += backbone_trainable

    params_ang = []
    for mod in ang_modules: params_ang += _params(mod)

    def _uniq(params):
        seen, out = set(), []
        for p in params:
            pid = id(p)
            if pid not in seen:
                seen.add(pid); out.append(p)
        return out

    params_kpt = _uniq(params_kpt)
    kpt_ids    = set(id(p) for p in params_kpt)
    params_ang = [p for p in _uniq(params_ang) if id(p) not in kpt_ids]

    all_trainable = [p for p in m.parameters() if p.requires_grad]
    if len(params_kpt) == 0 and len(params_ang) == 0 and len(all_trainable) > 0:
        params_kpt = all_trainable
    if len(params_kpt) == 0 and len(all_trainable) > 0:
        half = max(1, len(all_trainable)//2)
        params_kpt = all_trainable[:half]
        rest = [p for p in all_trainable if id(p) not in set(id(q) for q in params_kpt)]
        params_ang = rest
    if len(params_ang) == 0 and len(params_kpt) > 1:
        cut = max(1, len(params_kpt)//3)
        params_ang = params_kpt[-cut:]
        params_kpt = params_kpt[:-cut]
    elif len(params_ang) == 0 and len(params_kpt) > 0:
        params_ang = params_kpt[:1]

    optimizers = {
        'kpt': torch.optim.AdamW(params_kpt, lr=hyperparameters['lr_kpt']),
        'ang': torch.optim.AdamW(params_ang, lr=hyperparameters['lr_ang']),
    }

    warmup_epochs = hyperparameters.get('warmup_epochs', 5)
    total_epochs  = hyperparameters['num_epochs']
    schedulers = {
        'kpt': SequentialLR(
            optimizers['kpt'],
            [LinearLR(optimizers['kpt'], start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
             CosineAnnealingLR(optimizers['kpt'], T_max=total_epochs - warmup_epochs)],
            milestones=[warmup_epochs],
        ),
        'ang': SequentialLR(
            optimizers['ang'],
            [LinearLR(optimizers['ang'], start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
             CosineAnnealingLR(optimizers['ang'], T_max=total_epochs - warmup_epochs)],
            milestones=[warmup_epochs],
        ),
    }

    param_sets = {
        'kpt': set(id(p) for p in params_kpt),
        'ang': set(id(p) for p in params_ang),
    }

    if rank == 0:
        print(f"[setup] #params(kpt)={len(param_sets['kpt'])}, #params(ang)={len(param_sets['ang'])}, "
              f"total_trainable={sum(p.requires_grad for p in m.parameters())}")

    return model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler, param_sets, strong_transform
