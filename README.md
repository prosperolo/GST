# GST: Precise 3D Human Body from a Single Image with Gaussian Splatting Transformers

---------

## Installation
1. Create a conda environment 
    ```
    conda create --name gst python=3.10
    conda activate gst
    ```

2. Install [4DHumans](https://github.com/shubham-goel/4D-Humans) following the official instructions 

3. Install Gaussian splatting renderer [diff-gaussian-rasterization](https://github.com/ashawkey/diff-gaussian-rasterization)

4. Install other requirements
    ```
    pip install -r requirements.txt
    ```


## Training
```
python train_network.py +dataset=humman
```
