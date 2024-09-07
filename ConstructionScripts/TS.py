"""before use, you need to change the all file paths in this script"""

import pyrubberband as pyrb
import soundfile as sf


# 加载音频文件
def audiotimechange(src, dst, stretch_factor):
    y, sr = sf.read(src)

    # 设置时间拉伸的因子
    # stretch_factor = 1.5  # 拉伸1.5倍

    # 使用Rubber Band进行时间拉伸
    y_stretch = pyrb.time_stretch(y, sr, stretch_factor)

    # 保存拉伸后的音频
    sf.write(dst, y_stretch, sr)


def changetime(src, dst, stretch_factor):
    for root, dir, files in os.walk(src):
        os.makedirs(root.replace(src, dst), exist_ok=True)
        for file in files:
            if file.endswith(".wav"):
                try:
                    audiotimechange(
                        os.path.join(root, file),
                        os.path.join(root.replace(src, dst), file),
                        stretch_factor,
                    )
                except Exception as e:
                    print(f"Error: {e}")
            else:
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


src = "std.subset"
dst = ["TimeScaling90", "TimeScaling95", "TimeScaling105", "TimeScaling110"]
stretch_factor = [0.9, 0.95, 1.05, 110]
for i in range(len(dst)):
    changetime(src, dst[i], stretch_factor[i])
