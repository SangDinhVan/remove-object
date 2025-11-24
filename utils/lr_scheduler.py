import math
from torch.optim.lr_scheduler import LambdaLR

def lr_scheduler(optimizer, warmup_epochs=3, total_epochs=30, min_lr=1e-6, max_lr=1e-3):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Warmup: tăng tuyến tính từ min_lr → max_lr
            warmup_ratio = epoch / float(max(1, warmup_epochs))
            return (min_lr / max_lr) + (1 - min_lr / max_lr) * warmup_ratio
        
        # Cosine decay: giảm từ max_lr → 0
        progress = min(1.0, (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))  # từ 1 → 0

    scheduler = LambdaLR(optimizer, lr_lambda)
    return scheduler