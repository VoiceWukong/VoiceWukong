"""before use, you need to change the all file paths in this script"""

from pydub import AudioSegment
import math


def adjust_volume(input_file, output_file, target_ratio):
    """
    Adjust the volume of a WAV file to a target ratio.

    Args:
        input_file (str): Input WAV file path
        output_file (str): Output WAV file path
        target_ratio (float): Target volume ratio

    Returns:
        None
    """

    audio = AudioSegment.from_wav(input_file)

    gain = 20 * math.log10(target_ratio)

    adjusted_audio = audio.apply_gain(gain)

    adjusted_audio.export(output_file, format="wav")


def volume_change(src, dst, target_gain):
    for root, dir, files in os.walk(src):
        os.makedirs(root.replace(src, dst), exist_ok=True)
        for file in files:
            if file.endswith(".wav"):
                try:
                    adjust_volume(
                        os.path.join(root, file),
                        os.path.join(root.replace(src, dst), file),
                        target_gain,
                    )
                except Exception as e:

                    print(f"Error: {e}")
            else:
                shutil.copy(
                    os.path.join(root, file), os.path.join(root.replace(src, dst), file)
                )


src = "Std.subset"
dst = ["volume50", "volume75", "volume125", "volume150"]
target_gain = [0.5, 0.75, 1.25, 1.5]
for i in range(len(dst)):
    volume_change(src, dst[i], target_gain[i])
