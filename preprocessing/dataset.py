import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class InpaintDataset(Dataset):
    def __init__(self, root_dir, image_size=512,
                 extensions=(".jpg", ".jpeg", ".png"),
                 binary_mask=True):

        self.root_dir = root_dir

        # Tự động hiểu path
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        self.inpaint_dir = os.path.join(root_dir, "inpainted")

        self.extensions = tuple(extensions)
        self.binary_mask = binary_mask

        # Build danh sách sample
        self.samples = self._build_index()

        # Default transforms
        self.transform_img = T.Compose([
            T.Resize((image_size, image_size)),  # stretch resize; nếu muốn pad thì tớ viết thêm
            T.ToTensor(),
        ])
        self.transform_mask = T.Compose([
            T.Resize((image_size, image_size), interpolation=Image.NEAREST),
            T.ToTensor(),
        ])

    # ===== Factory từ YAML =====
    @classmethod
    def from_config(cls, cfg):
        return cls(
            root_dir=cfg["root_dir"],
            image_size=cfg.get("image_size", 512),
            extensions=tuple(cfg.get("extensions", [".jpg", ".jpeg", ".png"])),
            binary_mask=cfg.get("binary_mask", True),
        )

    # ===== Helper =====
    def _is_image(self, fn):
        return fn.lower().endswith(self.extensions)

    def _find_image(self, fid):
        """Tìm images/<fid>.*"""
        pattern = os.path.join(self.image_dir, f"{fid}*")
        files = [p for p in glob.glob(pattern) if self._is_image(p)]
        return files[0] if files else None

    # ===== Build index =====
    def _build_index(self):
        samples = []

        # duyệt tất cả folder trong masks (mỗi folder là 1 ID)
        for fid in os.listdir(self.mask_dir):
            mask_folder = os.path.join(self.mask_dir, fid)
            if not os.path.isdir(mask_folder):
                continue

            # tìm ảnh gốc
            img_path = self._find_image(fid)
            if img_path is None:
                continue

            # list mask
            mask_files = [f for f in os.listdir(mask_folder) if self._is_image(f)]
            if not mask_files:
                continue

            # folder inpaint tương ứng
            inpaint_folder = os.path.join(self.inpaint_dir, fid)
            if not os.path.isdir(inpaint_folder):
                continue

            # tạo sample cho từng mask
            for mf in mask_files:
                mask_path = os.path.join(mask_folder, mf)
                inpaint_path = os.path.join(inpaint_folder, mf)

                if not os.path.exists(inpaint_path):
                    continue

                samples.append({
                    "id": fid,
                    "image": img_path,
                    "mask": mask_path,
                    "inpaint": inpaint_path,
                    "mask_name": mf,
                })

        print(f"[InpaintDataset] Found {len(samples)} samples.")
        return samples

    # ===== PyTorch API =====
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        img = Image.open(item["image"]).convert("RGB")
        tgt = Image.open(item["inpaint"]).convert("RGB")
        mask = Image.open(item["mask"]).convert("L")

        img = self.transform_img(img)
        tgt = self.transform_img(tgt)
        mask = self.transform_mask(mask)

        if self.binary_mask:
            mask = (mask > 0.5).float()

        return {
            "id": item["id"],
            "mask_name": item["mask_name"],
            "image": img,
            "mask": mask,
            "target": tgt,
        }
