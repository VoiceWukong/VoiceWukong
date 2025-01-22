make sure python3.10.12 is installed
# Predict
- The prediction process can be completed using the ``eva.ipynb`` file
## Models
- download the weighted model [here](https://huggingface.co/VoiceWukong/VoiceWukong/resolve/main/OC-Softmax.pt?download=true)

- change "model_path" in the ``eva.ipynb`` file to the downloaded weighted model

## Dataset
- download the [part_aa](https://zenodo.org/records/13731918) and the [part_ab](https://zenodo.org/records/13732412) of **VoiceWukong** dataset, place both files in the same directory and use the following command to decompress
>>  ```cat VoiceWukong.part_* | tar -xz```

- put all *.m files in the same path

- change the "pathToDatabase" in the ``process_LA_data.m`` file to the path where **VoiceWukong** is located

- change the "pathToFeatures" in the ``process_LA_data.m`` file to save the features

- change the "evalProtocolFile" in the ``process_LA_data.m`` file to the path where ``eval_list.txt`` is located and execute the ``process_LA_data.m`` file with **Matlab**, you'll get English dataset LFCC_features in the ``Features`` path 

- change the "evalProtocolFile" in the ``process_LA_data.m`` file to the path where ``zh_eval_list.txt`` is located and execute the ``process_LA_data.m`` file with **Matlab**, you'll get English dataset LFCC_features in the ``ZH_Features`` path 


- change the "zh_features_path" and "en_features_path" in the ``eva.ipynb`` file to the path to "ZH_Features" and "Features" 

- update the save location for ```eval_score.txt``` within ``eva.ipynb`` to the desired folder for storing prediction results, and modify the associated file path to point to ``eval_list.txt``

## Run

- execute the cells in ``eva.ipynb`` file in order

# Analysis

- change the path to your **OC-Softmax** ``zh_eval_score.txt`` and ``en_eval_score.txt`` in ``analysis.ipynb`` file 

- execute the cells in ``analysis.ipynb`` file in order, you will get the same **Equal Error Rate (EER)** and **AUC score** as in the original paper

