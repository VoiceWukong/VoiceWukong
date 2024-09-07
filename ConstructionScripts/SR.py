"""before use, you need to change the all file paths in this script"""

import librosa
import soundfile as sf
import os
import shutil
import pandas as pd


def resample(src, dst, resr):
    for root, dir, files in os.walk(src):
        os.makedirs(root.replace(src, dst), exist_ok=True)
        for file in files:
            if file.endswith(".wav"):
                try:
                    y, sr = librosa.load(os.path.join(root, file), sr=None)
                    y_resample = librosa.resample(y, orig_sr=sr, target_sr=resr)

                    sf.write(
                        os.path.join(root.replace(src, dst), file), y_resample, resr
                    )
                except Exception as e:
                    if not os.path.exists("resampleerror.csv"):
                        df = pd.DataFrame(columns=["src_file", "dst_file", "error"])
                        df.to_csv("resampleerror.csv", index=False)
                    df = pd.read_csv("resampleerror.csv")
                    df.loc[len(df)] = [
                        os.path.join(root, file),
                        os.path.join(root.replace(src, dst), file),
                        str(e),
                    ]
                    # print(os.path.join(root, file))
                    print(f"Error: {e}")
            else:
                # os.makedirs(root.replace(src,dst),exist_ok=True)

                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


# resample('Wav-audio','Resample16K-audio')
src = "std.subset"
# dst=['/home/ydoit/AIGC/coqui-ai/xtts-Alldataset32K','/home/ydoit/AIGC/coqui-ai/xtts-Alldataset48k']
dst = ["Alldataset32K", "Alldataset44k"]
resr = [32000, 44100]
for i in range(len(dst)):
    resample(src, dst[i], resr[i])
