# IDO-VFI
This repository is an official implementation of IDO-VFI:
For academic use only.

****
# 👉Citation   

Citations are welcome, and if you use all or part of our codes in a published article or project, please cite the preprint version of our article that available on arXiv [IDO-VFI](https://arxiv.org/abs/2305.10198).

BibTeX of the preprint version:  
```
@misc{shi2023idovfi,  
title={IDO-VFI: Identifying Dynamics via Optical Flow Guidance for Video Frame Interpolation with Events},  
author={Chenyang Shi and Hanxiao Liu and Jing Jin and Wenzhuo Li and Yuzhen Li and Boyi Wei and Yibo Zhang},  
year={2023},  
eprint={2305.10198},  
archivePrefix={arXiv},  
primaryClass={cs.CV}  
}
```

****
# 💥Highlights

(1) A novel and trainable optical flow guidance mechanism for identifying the dynamics of the boundary frames and events is proposed, considering the corresponding relationship between adjacent dynamic regions.

(2) We propose an event-based residual optical flow estimation method to further dynamically evaluate the optical flow field, of which the computation time and computational effort are reduced by 10% and 17% respectively, while the performance is almost the same as processing the whole image without distinction.

(3) Our proposed method achieves state-of-the-art results on multiple benchmark datasets compared to frame-only and events-plus-frames VFI methods.

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

Download the checkpoint from here

The example data folder is organized as follows:

```
example
  ├── events_aligned
  |		├──000000.npz
  |		├──000001.npz
  |		...
  |
  └── images_corrected
  		├──timestamp.txt
  		├──000000.png
 		├──000001.png
 		...
```

To run IDO simply call

```
python --run_IDO.py --num_inter 5 --n_skip 0 --load_model ./hs_ergb.pth --data_path ./example
```

This will generate the output in `example/output`. The variables `n_skip` and `num_inter` determine the number of skipped and inserted frames. The variables  `load_model`  and `root_path` are  checkpoint model path and example data path. 

# Train

- Download the [Vimeo-90K](http://toflow.csail.mit.edu/) dataset 
- Generate  synthetic events using [ESIM](https://rpg.ifi.uzh.ch/docs/CORL18_Rebecq.pdf) . The python bindings for ESIM can be found in [rpg_vid2e](https://github.com/uzh-rpg/rpg_vid2e). 

- Then train the model and training details can be seen in paper. 

  ```
  python train.py --dataset_path <dataset_path>
  ```

  Or you can directly use the example data of  Vimeo-90K to run the training code.

  ```
  python train.py --dataset_path ./dataset/vimeo_triplet
  ```

# Test

After training, you can evaluate the model on Vimeo-90k dataset following command,

```
python --test.py --n_skip 1 --load_model <checkpoint_path> --data_path <dataset_path> --eval_dataset triplet_test
```

You can also evaluate IDO using our weight [here](https://drive.google.com/drive/folders/10m6RNhWeaEiDbZxgMYZAAkLIeLxrUkuN?usp=share_link) on HighQualityFrames(HQF) dataset. You can download the High Quality Frames dataset in HDF5 format from [here](https://drive.google.com/drive/folders/1s5JF2Pt4lgFr_x0B3WvCRpAVOob2Y1vn?usp=sharing) . Or you can go to  [the official website](https://github.com/coco-tasks/dataset) to download the dataset and convert it to HDF5 format following the instruction in the official website. 

```
python --test.py --n_skip 1 --load_model ./Hqf.pth --data_path <dataset_path> --eval_dataset Hqf
```

The variable `n_skip` determine the number of skipped frames. 

# Reference

Some other great video interpolation resources that we benefit from:

[Video-Frame-Interpolation-Transformer](https://github.com/zhshi0816/Video-Frame-Interpolation-Transformer)

[TimeLens](https://github.com/uzh-rpg/rpg_timelens)

If you use our entire codes, please cite the methods above as well.
