# IDO-VFI
This repository is an official implementation of IDO-VFI:
For academic use only.

****
👉Citation   
Citations are welcome, and if you use all or part of our code in a published article or project, please cite the preprint version of our article that available on arXiv [IDO-VFI](https://arxiv.org/abs/2305.10198).

BibTeX of the preprint version:  
@misc{shi2023idovfi,  
      title={IDO-VFI: Identifying Dynamics via Optical Flow Guidance for Video Frame Interpolation with Events},  
      author={Chenyang Shi and Hanxiao Liu and Jing Jin and Wenzhuo Li and Yuzhen Li and Boyi Wei and Yibo Zhang},  
      year={2023},  
      eprint={2305.10198},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV}  
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

# Reference

Some other great video interpolation resources that we benefit from:

[VFIT-B](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer)

[TimeLens](https://github.com/uzh-rpg/rpg_timelens)
