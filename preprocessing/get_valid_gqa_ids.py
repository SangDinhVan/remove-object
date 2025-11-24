import zipfile
import os

def get_valid_gqa_ids(GQA_IMAGES_ZIP, GQA_INPAINT_ZIP, limit=None):

    def is_int(s: str) -> bool:
        try:
            int(s)
            return True
        except ValueError:
            return False

    # ====== Đọc ảnh gốc trong images.zip ======
    with zipfile.ZipFile(GQA_IMAGES_ZIP, 'r') as zimg:
        img_files = [n for n in zimg.namelist()
                     if n.lower().endswith((".jpg", ".jpeg", ".png"))]

        image_ids = set()
        for f in img_files:
            stem = os.path.splitext(os.path.basename(f))[0]  # tên file
            if is_int(stem):
                image_ids.add(stem)

    # ====== Đọc masks + inpainted ======
    with zipfile.ZipFile(GQA_INPAINT_ZIP, 'r') as zinp:
        names_inp = zinp.namelist()

        mask_ids = set()
        inpaint_ids = set()

        for n in names_inp:
            if n.startswith("masks/") and n.lower().endswith((".png",".jpg",".jpeg")):
                parts = n.split("/")
                if len(parts) >= 3:
                    fid = parts[1]
                    if is_int(fid):
                        mask_ids.add(fid)

            elif n.startswith("images_inpainted/") and n.lower().endswith((".png",".jpg",".jpeg")):
                parts = n.split("/")
                if len(parts) >= 3:
                    fid = parts[1]
                    if is_int(fid):
                        inpaint_ids.add(fid)

    # ====== Tìm ID hợp lệ: có hình gốc + mask + inpaint ======
    valid_ids = image_ids & mask_ids & inpaint_ids
    valid_ids_sorted = sorted(valid_ids, key=lambda x: int(x))
    # return list sort theo số
    if limit is not None:
        return valid_ids_sorted[:limit]

    return valid_ids_sorted
