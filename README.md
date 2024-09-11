# GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers

---------

## Installation
1. Create a conda environment 
    ```
    conda create --name gst python=3.10
    conda activate gst
    ```

2. Install [4DHumans](https://github.com/shubham-goel/4D-Humans) following the official instructions. Download the SMPL model according to the repo README instructions, then run the `demo.py` script to download the pretrained models and store them under the correct path 

3. Install Gaussian splatting renderer [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization)

4. Install [simple-knn](https://gitlab.inria.fr/bkerbl/simple-knn) 

5. Install other requirements
    ```
    pip install -r requirements.txt
    ```

## Data preparation
To train on the [HuMMan](https://caizhongang.com/projects/HuMMan/recon.html) dataset 
1. Download the dataset following the official instruction 
2. Unzip the data and change the `HUMMAN_DATASET_ROOT` to the unzipped folder inside `scene/humman.py`


## Training
```
python train_network.py +dataset=humman
```
