
"""before use, you need to change the all file paths in this script"""
import numpy as np
import soundfile as sf
import os
import shutil
import pandas as pd

def linear_fade_in_out(input_file, output_file, ratio):
    """
    Apply linear fade-in and fade-out to a WAV file.

    Args:
        input_file (str): Input WAV file path
        output_file (str): Output WAV file path
        ratio (float): Fade-in and fade-out ratio, relative to the total duration of the audio

    Returns:
        None
    """

    data, rate = sf.read(input_file)

    fade_samples = int(len(data) * ratio)

    fade_in = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)

    data[:fade_samples] = data[:fade_samples] * fade_in
    data[-fade_samples:] = data[-fade_samples:] * fade_out

    sf.write(output_file, data, rate)




def exponential_fade_in_out(input_file, output_file, ratio):
    """
    Apply exponential fade-in and fade-out to a WAV file.

    Args:
        input_file (str): Input WAV file path
        output_file (str): Output WAV file path
        fade_in_ratio (float): Fade-in ratio, relative to the total duration of the audio
        fade_out_ratio (float): Fade-out ratio, relative to the total duration of the audio

    Returns:
        None
    """

    data, rate = sf.read(input_file)

    fade_in_samples = int(len(data) * ratio)
    fade_out_samples = int(len(data) * ratio)

    fade_in = np.linspace(0.0, 1.0, fade_in_samples) ** 2
    fade_out = np.linspace(1.0, 0.0, fade_out_samples) ** 2

    data[:fade_in_samples] = data[:fade_in_samples] * fade_in
    data[-fade_out_samples:] = data[-fade_out_samples:] * fade_out

    sf.write(output_file, data, rate)



def logarithmic_fade_in_out(input_file, output_file, ratio):
    """
    Apply logarithmic fade-in and fade-out to a WAV file.
    
    Args:
        input_file (str): Input WAV file path
        output_file (str): Output WAV file path
        ratio (float): Fade-in and fade-out ratio, relative to the total duration of the audio
        
    Returns:
        None
    """

    data, rate = sf.read(input_file)
    fade_samples = int(len(data) * ratio)

    fade_in = np.logspace(-3, 0, fade_samples)
    fade_out = np.logspace(0, -3, fade_samples)

    data[:fade_samples] = data[:fade_samples] * fade_in
    data[-fade_samples:] = data[-fade_samples:] * fade_out

    sf.write(output_file, data, rate)

def linefade(src,dst,ratio):
    for root,dir,files in os.walk(src):
        os.makedirs(root.replace(src,dst),exist_ok=True)
        for file in files:
            if file.endswith('.wav'):
                try:
                    linear_fade_in_out(os.path.join(root, file),os.path.join(root.replace(src,dst), file),ratio)
                except Exception as e:
                    if not os.path.exists('linefadeerror.csv'):
                        df=pd.DataFrame(columns=['src_file','dst_file','error'])
                        df.to_csv('linefadeerror.csv',index=False)
                    df=pd.read_csv('linefadeerror.csv')
                    df.loc[len(df)]=[os.path.join(root, file),os.path.join(root.replace(src,dst), file),e] 
                    df.to_csv('linefadeerror.csv',index=False)
                    print(f'Error: {e}')
            else:
                shutil.copy(os.path.join(root, file),os.path.join(root.replace(src,dst), file))
def expfade(src,dst,ratio):
    for root,dir,files in os.walk(src):
        os.makedirs(root.replace(src,dst),exist_ok=True)
        for file in files:
            if file.endswith('.wav'):
                try :
                    exponential_fade_in_out(os.path.join(root, file),os.path.join(root.replace(src,dst), file),ratio)
                except Exception as e:
                    if not os.path.exists('expfadeerror.csv'):
                        df=pd.DataFrame(columns=['src_file','dst_file','error'])
                        df.to_csv('expfadeerror.csv',index=False)
                    df=pd.read_csv('expfadeerror.csv')
                    df.loc[len(df)]=[os.path.join(root, file),os.path.join(root.replace(src,dst), file),e]
                    df.to_csv('expfadeerror.csv',index=False)
                    print(f'Error: {e}')
            else:
                shutil.copy(os.path.join(root, file),os.path.join(root.replace(src,dst), file))
def logfade(src,dst,ratio):
    for root,dir,files in os.walk(src):
        os.makedirs(root.replace(src,dst),exist_ok=True)
        for file in files:
            if file.endswith('.wav'):
                try:
                    logarithmic_fade_in_out(os.path.join(root, file),os.path.join(root.replace(src,dst), file),ratio)
                except Exception as e:
                    if not os.path.exists('logfadeerror.csv'):
                        df=pd.DataFrame(columns=['src_file','dst_file','error'])
                        df.to_csv('logfadeerror.csv',index=False)
                    df=pd.read_csv('logfadeerror.csv')
                    df.loc[len(df)]=[os.path.join(root, file),os.path.join(root.replace(src,dst), file),e]
                    df.to_csv('logfadeerror.csv',index=False)
                    print(f'Error: {e}')
            else:
                shutil.copy(os.path.join(root, file),os.path.join(root.replace(src,dst), file))
                
src='std.subset'
dsts=['linefade',
     'expfade',
     'logfade']
ratios=[0.1,0.2,0.3]
for ratio in ratios:
    for dst in dsts:
        if 'linefade' in dst:
            linefade(src,f'{dst}{int(ratio*100)}',ratio)
        elif 'expfade' in dst:
            expfade(src,f'{dst}{int(ratio*100)}',ratio)
        else:
            logfade(src,f'{dst}{int(ratio*100)}',ratio)
