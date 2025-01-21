make sure python3.10.12 is installed
# Predict
- The prediction process can be completed using the ``eva.ipynb`` file
## Models
- download the weighted model [here](https://huggingface.co/VoiceWukong/VoiceWukong/resolve/main/RawPCDARTS.pth?download=true)

- change "model_path" in the ``eva.ipynb`` file to the downloaded weighted model

## Dataset
- download the [part_aa](https://zenodo.org/records/13731918) and the [part_ab](https://zenodo.org/records/13732412) of **VoiceWukong** dataset, place both files in the same directory and use the following command to decompress
>>  ```cat VoiceWukong.part_* | tar -xz```

- change the path of the corresponding dataset in the ```eva.ipynb``` file to the path of the decompressed data set

- modify the associated file path to point to ``eval_list.txt`` in the ``eva.ipnb`` file

## Run

- execute the cells in ``eva.ipynb`` file in order

# Analysis

- change the path to your **RawPCDARTS** ``zh_eval_score.txt`` and ``en_eval_score.txt`` in ``analysis.ipynb`` file 

- execute the cells in ``analysis.ipynb`` file in order, you will get the same **Equal Error Rate (EER)** and **AUC score** as in the original paper
