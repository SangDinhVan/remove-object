import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np

# import inference utilities
from inference import load_config, build_model_from_config, inpaint_pil


class InpaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Removal Inpainting GUI")

        # Ảnh & mask
        self.original_image = None      # PIL RGB - ảnh gốc (đã resize cho GUI)
        self.display_image = None       # PIL RGB - ảnh hiển thị bên trái
        self.result_image = None        # PIL RGB - ảnh inpaint kết quả

        self.mask_image = None          # PIL L (0..255) - mask vẽ lên
        self.mask_draw = None           # ImageDraw để vẽ lên mask

        self.brush_size = tk.IntVar(value=20)
        self.is_painting = False

        # ===== Load config + model 1 lần =====
        try:
            self.cfg = load_config("config/config.yaml")
            ckpt_path = "output/checkpoints/best_model2.pth"
            self.model, self.cfg, self.device = build_model_from_config(
                self.cfg, ckpt_path
            )
            print(f"[INFO] Model loaded on {self.device}")
        except Exception as e:
            messagebox.showerror("Lỗi load model", f"{e}")
            self.model = None
            self.device = "cpu"
            self.cfg = None

        # ===== Build UI =====
        self.tk_original = None
        self.tk_result = None
        self.build_ui()

    # ===================== UI =====================
    def build_ui(self):
        # Thanh control phía trên
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        btn_load = tk.Button(control_frame, text="Chọn ảnh", command=self.load_image)
        btn_load.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Cỡ bút:").pack(side=tk.LEFT)
        scale_brush = tk.Scale(
            control_frame,
            from_=5,
            to=80,
            orient=tk.HORIZONTAL,
            variable=self.brush_size,
        )
        scale_brush.pack(side=tk.LEFT, padx=5)

        btn_clear = tk.Button(control_frame, text="Xóa mask", command=self.clear_mask)
        btn_clear.pack(side=tk.LEFT, padx=5)

        btn_inpaint = tk.Button(control_frame, text="Inpaint (model)", command=self.run_inpaint)
        btn_inpaint.pack(side=tk.LEFT, padx=5)

        # Khung chính chia đôi
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Bên trái: ảnh gốc + mask
        left_frame = tk.LabelFrame(main_frame, text="Ảnh gốc + vùng bôi")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_original = tk.Canvas(left_frame, bg="gray")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # Bên phải: ảnh inpaint
        right_frame = tk.LabelFrame(main_frame, text="Ảnh đã inpaint")
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.canvas_result = tk.Canvas(right_frame, bg="gray")
        self.canvas_result.pack(fill=tk.BOTH, expand=True)

        # Bind event vẽ mask
        self.canvas_original.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas_original.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas_original.bind("<ButtonRelease-1>", self.on_mouse_up)

    # ===================== Image & Mask =====================
    def load_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.tif")]
        )
        if not file_path:
            return

        img = Image.open(file_path).convert("RGB")

        # Resize vừa GUI (giữ tỉ lệ)
        max_size = 768
        w, h = img.size
        scale = min(max_size / max(w, h), 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        self.original_image = img
        self.display_image = img.copy()

        # Mask trắng (0) ban đầu
        self.mask_image = Image.new("L", self.display_image.size, 0)
        self.mask_draw = ImageDraw.Draw(self.mask_image)

        self.result_image = None

        self.update_original_canvas()
        self.update_result_canvas()

    def clear_mask(self):
        if self.display_image is None:
            return
        self.mask_image = Image.new("L", self.display_image.size, 0)
        self.mask_draw = ImageDraw.Draw(self.mask_image)
        self.update_original_canvas()

    def update_original_canvas(self):
        self.canvas_original.delete("all")
        if self.display_image is None:
            return

        # Tạo overlay đỏ theo mask
        overlay = self.display_image.copy()
        overlay = overlay.convert("RGBA")

        mask_colored = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
        # alpha = 128 nếu mask > 0
        alpha_mask = self.mask_image.point(lambda x: 128 if x > 0 else 0)
        mask_colored.putalpha(alpha_mask)

        overlay = Image.alpha_composite(overlay, mask_colored).convert("RGB")

        self.tk_original = ImageTk.PhotoImage(overlay)
        self.canvas_original.config(width=overlay.width, height=overlay.height)
        self.canvas_original.create_image(0, 0, anchor=tk.NW, image=self.tk_original)

    def update_result_canvas(self):
        self.canvas_result.delete("all")
        if self.result_image is None:
            return
        self.tk_result = ImageTk.PhotoImage(self.result_image)
        self.canvas_result.config(width=self.result_image.width, height=self.result_image.height)
        self.canvas_result.create_image(0, 0, anchor=tk.NW, image=self.tk_result)

    # ===================== Mouse drawing =====================
    def on_mouse_down(self, event):
        if self.display_image is None:
            return
        self.is_painting = True
        self.paint_at(event.x, event.y)

    def on_mouse_move(self, event):
        if not self.is_painting or self.display_image is None:
            return
        self.paint_at(event.x, event.y)

    def on_mouse_up(self, event):
        self.is_painting = False

    def paint_at(self, x, y):
        """Vẽ chấm tròn lên mask & cập nhật overlay."""
        if self.mask_draw is None:
            return

        r = self.brush_size.get() // 2
        left, top = x - r, y - r
        right, bottom = x + r, y + r

        # vẽ vùng trắng (255) lên mask
        self.mask_draw.ellipse([left, top, right, bottom], fill=255)
        self.update_original_canvas()

    # ===================== Inpaint bằng model =====================
    def run_inpaint(self):
        if self.display_image is None or self.mask_image is None:
            messagebox.showwarning("Chưa có ảnh", "Hãy chọn ảnh trước đã.")
            return

        if self.model is None or self.cfg is None:
            messagebox.showerror("Lỗi model", "Model chưa load được.")
            return

        try:
            # Gọi hàm inpaint_pil từ inference.py
            out_pil = inpaint_pil(
                self.model,
                self.cfg,
                self.device,
                self.display_image,
                self.mask_image,
            )
            self.result_image = out_pil
            self.update_result_canvas()
        except Exception as e:
            messagebox.showerror("Lỗi inpaint", f"{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = InpaintApp(root)
    root.mainloop()
