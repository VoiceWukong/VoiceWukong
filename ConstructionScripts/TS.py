"""before use, you need to change the all file paths in this script"""

import pyrubberband as pyrb
import soundfile as sf



def audiotimechange(src, dst, stretch_factor):
    y, sr = sf.read(src)

 
    y_stretch = pyrb.time_stretch(y, sr, stretch_factor)


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
