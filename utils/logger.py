import os
import logging

def setup_logger(save_dir, log_name="train"):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"{log_name}.log")

    logger = logging.getLogger(log_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # tránh trùng handler nếu gọi lại

    # Ghi ra file (append mode) 
    file_handler = logging.FileHandler(log_file, mode='a')  # 'a' = append, không overwrite
    file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))

    # Ghi ra console 
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    if os.path.getsize(log_file) == 0:
        logger.info("epoch,train_loss,train_f1,train_auc,train_pr_auc,val_loss,val_f1,val_auc,val_pr_auc")

    return logger
