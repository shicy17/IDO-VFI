# IDO-VFI
This repository is an official implementation of IDO-VFI:\
For academic use only.

****
👉Citation   
Citations are welcome, and if you use all or part of our code in a published article or project, please cite the preprint version of our article that available on arXiv.

bibTeX of the preprint version:  
@misc{XXXX,  
Author = {},  
Title = {},  
Year = {2023},  
Eprint = {},  
}  
****
💥Highlights

****

# Installation

Install the dependencies with

```
conda create -y -n IDO_VFI python=3.8
conda activate IDO_VFI
pip install -r requirements.txt
```



****

# Running

**Download data**

Download the High Quality Frames dataset in HDF5 format from <u>dataset_link</u>. Or you can go to  [the official website](https://github.com/coco-tasks/dataset) to download the High Quality Frames dataset and convert it to HDF5 format following the instruction in the official website. 

You can organize the 'data' folder as follows:

```
data/
  ├── Hqfdataset
       ├── bike_bay_hdr.h5
       ...
       └── still_life.h5
```

**Download pretrained models**

We provide our pretrained models on <u>model_link</u>.

**Evaluation**

```
python --test.py --eval_dataset Hqf --n_skip 1  --load_model ./Hqf.pth 
```

****

# 
