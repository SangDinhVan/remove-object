import os
import zipfile
import shutil
from get_valid_gqa_ids import get_valid_gqa_ids
# ================================
GQA_IMAGES_ZIP   = r"S:\Downloads\images.zip"
GQA_INPAINT_ZIP  = r"S:\Downloads\gqa-inpaint.zip"
OUT              = r"S:\Downloads\sample_id_1_20"
# ================================

os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
os.makedirs(os.path.join(OUT, "masks"), exist_ok=True)
os.makedirs(os.path.join(OUT, "inpainted"), exist_ok=True)

# base_ids = [str(i) for i in range(1, 10000)]
base_ids = get_valid_gqa_ids(GQA_IMAGES_ZIP, GQA_INPAINT_ZIP, 10000)
print("Đang lấy 20 ID:", base_ids)

# ==== 1) MAP ID → ảnh gốc ====
print("Đang đọc:", GQA_IMAGES_ZIP)
with zipfile.ZipFile(GQA_IMAGES_ZIP, 'r') as zimg:
    img_files = [n for n in zimg.namelist() if n.lower().endswith((".jpg",".png"))]

    id2img_path = {}
    for f in img_files:
        stem = os.path.splitext(os.path.basename(f))[0]
        if stem in base_ids:
            id2img_path[stem] = f

print("Ảnh gốc tìm được:", id2img_path.keys())

# ==== 2) XỬ LÝ ====
print("Đang đọc:", GQA_INPAINT_ZIP)
with zipfile.ZipFile(GQA_INPAINT_ZIP, 'r') as zinp:
    names_inp = zinp.namelist()

    for fid in base_ids:
        print(f"\n=== Xử lý ID {fid} ===")

        # ----- ẢNH GỐC -----
        if fid not in id2img_path:
            print(f"⚠ Không có ảnh gốc cho ID {fid}, bỏ qua.")
            continue

        img_member = id2img_path[fid]
        dst_img = os.path.join(OUT, "images", f"{fid}.jpg")

        # chép ảnh gốc tạm
        with zipfile.ZipFile(GQA_IMAGES_ZIP, 'r') as zimg:
            with zimg.open(img_member) as src, open(dst_img, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # ----- MASK -----
        mask_prefix = f"masks/{fid}/"
        mask_members = [
            n for n in names_inp
            if n.startswith(mask_prefix) and n.lower().endswith((".png",".jpg",".jpeg"))
        ]
        has_mask = len(mask_members) > 0

        # ----- INPAINTED -----
        inp_prefix = f"images_inpainted/{fid}/"
        inp_members = [
            n for n in names_inp
            if n.startswith(inp_prefix) and n.lower().endswith((".png",".jpg",".jpeg"))
        ]
        has_inpainted = len(inp_members) > 0

        # ----- CHECK ĐỦ 3 LOẠI -----
        if not has_mask or not has_inpainted:
            print(f"❌ ID {fid} không đủ dữ liệu → XÓA ảnh gốc")

            # xóa ảnh gốc nếu có
            if os.path.exists(dst_img):
                os.remove(dst_img)

            continue  # bỏ qua không chép mask/inpaint

        # ======= ĐẾN ĐÂY: ĐỦ 3 LOẠI =======
        print(f"✅ ID {fid} hợp lệ → lưu mask + inpaint")

        # lưu masks
        out_mask_dir = os.path.join(OUT, "masks", fid)
        os.makedirs(out_mask_dir, exist_ok=True)
        for m in mask_members:
            dst_mask = os.path.join(out_mask_dir, os.path.basename(m))
            with zinp.open(m) as src, open(dst_mask, "wb") as dst:
                shutil.copyfileobj(src, dst)

        # lưu inpainted
        out_inp_dir = os.path.join(OUT, "inpainted", fid)
        os.makedirs(out_inp_dir, exist_ok=True)
        for m in inp_members:
            dst_inp = os.path.join(out_inp_dir, os.path.basename(m))
            with zinp.open(m) as src, open(dst_inp, "wb") as dst:
                shutil.copyfileobj(src, dst)

print("\n🎉 DONE! Dữ liệu sample nằm tại:", OUT)


