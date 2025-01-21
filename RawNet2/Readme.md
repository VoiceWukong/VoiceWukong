make sure python3.10.12 is installed
# Predict
- The prediction process can be completed using the ``eva.ipynb`` file
## Models
- download the weighted model [here](https://huggingface.co/VoiceWukong/VoiceWukong/resolve/main/rawnet2.pth?download=true)

- change "model_path" in the ``eva.ipynb`` file to the downloaded weighted model

## Dataset
- download the [part_aa]() and the [part_ab]() of **VoiceWukong** dataset, place both files in the same directory and use the following command to decompress
>>  ```cat VoiceWukong.part_* | tar -xz```

- change the path of the corresponding dataset in the ```eva.ipynb``` file to the path of the decompressed data set

- update the save location for ```eval_score.txt``` within ``eva.ipynb`` to the desired folder for storing prediction results, and modify the associated file path to point to ``eval_list.txt``

## Run

- execute the cells in ``eva.ipynb`` file in order

# Analysis

- change the path to your **RawNet2** ``zh_eval_score.txt`` and ``en_eval_score.txt`` in ``analysis.ipynb`` file 

- execute the cells in ``analysis.ipynb`` file in order, you will get the same **Equal Error Rate (EER)** and **AUC score** as in the original paper
