import os
import json
import numpy as np
import cv2

# === 配置路径 ===
input_dir = r"D:\test\Source1"
output_dir = r"D:\test\Source1_RGB"
os.makedirs(output_dir, exist_ok=True)

# === 支持的像素类型 ===
PIXEL_TYPE_MONO8 = 0x01080001
PIXEL_TYPE_BAYERRG8 = 0x01080009  # 17301513

# === 提取 block_id（用于排序） ===
def extract_blockid(filename):
    try:
        return int(filename.split("_")[1])  # 提取中间的 X 并转为整数
    except (IndexError, ValueError):
        return -1  # 出错时默认排序在最前

# === 获取所有 .bin 文件，并按 block_id 排序 ===
bin_files = sorted(
    [f for f in os.listdir(input_dir) if f.endswith(".bin")],
    key=extract_blockid
)

print(f"Found {len(bin_files)} frames")

# === 播放并保存图像循环 ===
for bin_file in bin_files:
    bin_path = os.path.join(input_dir, bin_file)
    meta_path = bin_path.replace(".bin", ".json")

    if not os.path.exists(meta_path):
        print(f"[Missing meta] {meta_path}")
        continue

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    width = meta["width"]
    height = meta["height"]
    pixel_type = meta["pixel_type"]
    block_id = meta.get("block_id", "unknown")

    with open(bin_path, 'rb') as f:
        raw = f.read()

    image = None
    if pixel_type == PIXEL_TYPE_MONO8:
        image = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        save_path = os.path.join(output_dir, f"frame_{block_id}.bmp")
        cv2.imwrite(save_path, image)
    elif pixel_type == PIXEL_TYPE_BAYERRG8:
        bayer = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
        image = cv2.cvtColor(bayer, cv2.COLOR_BayerRG2RGB)
        save_path = os.path.join(output_dir, f"frame_{block_id}.bmp")
        # cv2.imwrite(save_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))  # 保存为 BGR 格式
        cv2.imwrite(save_path, image)
    else:
        print(f"[Unsupported pixel type] {pixel_type}")
        continue

    # 显示窗口
    cv2.namedWindow("Playback", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Playback", 800, 600)
    cv2.imshow("Playback", image)

    key = cv2.waitKey(30)
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
print("Finished playback and saving.")
