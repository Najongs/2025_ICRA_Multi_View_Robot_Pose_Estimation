# setup.py (또는 학습 스크립트 상단)
import itertools
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
import torchvision.transforms as transforms

# 통합 Dataset/Utils/Loss
from dataset import UnifiedRobotPoseDataset          # 앞서 만든 통합본
from utils import MAX_VIEWS_PER_GROUP                # 필요시 사용
from loss_and_metrics import make_angle_loss         # 네가 정의한 함수 사용

# ---------------------------------------------------------
# 공용 collate: dict-of-views 배치 패딩
# ---------------------------------------------------------
def make_collate_pad_dicts():
    def collate_fn(batch):
        # (None,None,None) 제거
        batch = [b for b in batch if b[0] is not None]
        if not batch:
            return None, None, None

        image_dicts, heatmap_dicts, angles_list = zip(*batch)

        # 배치 내 모든 키 수집
        all_keys = sorted(list(set(itertools.chain.from_iterable(d.keys() for d in image_dicts))))

        # 더미 텐서 준비
        sample_img_tensor = next(iter(image_dicts[0].values()))
        sample_hmap_tensor = next(iter(heatmap_dicts[0].values()))
        dummy_img = torch.zeros_like(sample_img_tensor)
        dummy_hmap = torch.zeros_like(sample_hmap_tensor)

        padded_images, padded_heatmaps = [], []
        for i in range(len(batch)):
            padded_img_dict = {k: image_dicts[i].get(k, dummy_img) for k in all_keys}
            padded_hmap_dict = {k: heatmap_dicts[i].get(k, dummy_hmap) for k in all_keys}
            padded_images.append(padded_img_dict)
            padded_heatmaps.append(padded_hmap_dict)

        images_collated   = torch.utils.data.dataloader.default_collate(padded_images)    # {vk: (B,C,H,W)}
        heatmaps_collated = torch.utils.data.dataloader.default_collate(padded_heatmaps)  # {vk: (B,J,Hh,Wh)}
        angles_collated   = torch.stack(angles_list)

        return images_collated, heatmaps_collated, angles_collated
    return collate_fn

# ---------------------------------------------------------
# Experiment Setup (통합 버전)
# ---------------------------------------------------------
def setup(dataset_type,            # 'fr3' | 'fr5' | 'meca500'
          dataset_items,           # groups or pairs (UnifiedRobotPoseDataset가 바로 받는 포맷)
          hyperparameters,
          rank, world_size,
          model_cls,               # 예: DINOv3PoseEstimator
          extra_model_kwargs=None  # 필요시 {'model_name': ..., ...}
          ):
    print(f"--- [Rank {rank}] Setting up environment for {dataset_type.upper()} ---")
    device = torch.device(f"cuda:{rank}")

    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    resize_size = hyperparameters.get("input_size", 224)
    heatmap_size = hyperparameters.get("heatmap_size", (128,128))
    sigma = hyperparameters.get("sigma", 5.0)

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
    n_train = int(len(dataset_items) * (1 - hyperparameters['val_split']))
    train_items = [dataset_items[i] for i in idx[:n_train]]
    val_items   = [dataset_items[i] for i in idx[n_train:]]

    # 멀티뷰만 2뷰 이상 필터링(싱글뷰 페어는 그대로 유지)
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
        robot=dataset_type,          # FK 대상 로봇 = 데이터셋 타입
        robot_fk_unit=None,          # 각 스펙의 기본 단위(rad/deg) 자동 사용
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

    # --------- num_angles 자동 설정 ----------
    # 1) 로봇 규칙 우선
    default_angles = {'fr3': 7, 'fr5': 6, 'meca500': 6}
    num_angles = default_angles.get(dataset_type, None)
    # 2) 데이터에서 확인(안전망)
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

    # --------- model ----------
    extra_model_kwargs = extra_model_kwargs or {}
    model = model_cls(**extra_model_kwargs).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # --------- losses / criteria ----------
    angle_loss = make_angle_loss(num_angles, vm_weight=0.5, cos_weight=0.5)
    if hasattr(angle_loss, "vm"):
        angle_loss.vm = angle_loss.vm.to(device)

    # (선택) FK regularizer가 있다면 가져오기
    fk_reg = None
    try:
        from loss_and_metrics import FKRegularizer  # 있다면 쓰기
        fk_reg = FKRegularizer(robot=dataset_type, device=device)
    except Exception:
        fk_reg = None

    criteria = {'ang': angle_loss}
    if fk_reg is not None:
        criteria['fk'] = fk_reg
        criteria['lambda_fk'] = hyperparameters.get('lambda_fk', 0.5)
    else:
        criteria['lambda_fk'] = 0.0

    # --------- optimizer / scheduler ----------
    m = model.module
    # 안전하게 속성 체크해서 파라미터 모으기
    def params_of(name):
        return list(getattr(m, name).parameters()) if hasattr(m, name) else []

    params_shared = params_of('view_embeddings') + params_of('fusion')
    params_ang = params_of('ang_head') + params_shared + params_of('kp_token_enc') + params_of('cnn_token_enc')
    # angle_loss 내부에 학습가능 파라미터(von-Mises 등)가 있으면 추가
    if hasattr(angle_loss, "vm"):
        try: params_ang += list(angle_loss.vm.parameters())
        except Exception: pass

    params_kpt = params_of('cnn_stem') + params_shared + params_of('kpt_enricher') + params_of('kpt_head')

    optimizers = {
        'kpt': torch.optim.AdamW(params_kpt, lr=hyperparameters['lr_kpt']),
        'ang': torch.optim.AdamW(params_ang, lr=hyperparameters['lr_ang']),
    }

    warmup_epochs = hyperparameters.get('warmup_epochs', 5)
    total_epochs  = hyperparameters['num_epochs']
    schedulers = {
        'kpt': SequentialLR(
            optimizers['kpt'],
            [
                LinearLR(optimizers['kpt'], start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizers['kpt'], T_max=max(total_epochs - warmup_epochs, 1)),
            ],
            milestones=[warmup_epochs]
        ),
        'ang': SequentialLR(
            optimizers['ang'],
            [
                LinearLR(optimizers['ang'], start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizers['ang'], T_max=max(total_epochs - warmup_epochs, 1)),
            ],
            milestones=[warmup_epochs]
        ),
    }

    # (선택) param_sets: parameter grouping을 추적하고 싶다면 id로 집합 구성
    param_sets = {
        'kpt': set(id(p) for p in params_kpt),
        'ang': set(id(p) for p in params_ang),
    }

    return model, train_loader, val_loader, criteria, optimizers, schedulers, device, mean, std, train_sampler, param_sets, strong_transform
