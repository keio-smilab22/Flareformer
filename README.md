# Flareformer

PyTorch training code for **Flareformer**.

![model](https://user-images.githubusercontent.com/51681991/174759454-ae57f8e8-f0ed-4b75-b976-7099f69e1bdc.png)

## Getting Started

### Requirements

- Python >= 3.8.10
- CUDA >= 11.1
- PyTorch >= 1.8.2

### Installation

1. `git clone git@github.com:keio-smilab22/Flareformer.git`
2. `cd Flareformer`
3. `pip install -r requirements.txt`


## Data preparation

- The required data files should be put under ```data/``` folder.
- Visit http://wdc.nict.go.jp/IONO/wdc/solarflare/index.html and download the database of physical features.

```
$ cd ~/work/flare_transformer
$ mv ~/data.zip data/
$ unzip data/data.zip
```

- Visit https://sdo.gsfc.nasa.gov/data/ and download hourly magnetograms.
```
$ mv ~/magnetogram_images.tar.gz data/
$ tar -zxvf data/magnetogram_images.tar.gz
```

## Preprocess

- Preprocess the physical features and magnetogram images by the following procedure (parallel processing).

    - `python preprocess/main.py --magnetogram --physical --label --window`

- The following data files should be created under ```data/```.
  -  data/data_20XX_magnetogram_256.npy
  -  data/data_20XX_feat.csv
  -  data/data_20XX_label.csv
  -  data/data_20XX_window_48.csv

## Training

- To train Flareformer with warmup and [cRT](https://arxiv.org/abs/1910.09217) using `params/params_2014.json`: 

    - `python main.py --params params/params_2014.json --warmup_epochs=5 --imbalance --wandb`

## License

Flareformer is released under the BSD-3-Clause-Clear license. Please see the [LICENSE](LICENSE) file for more information.

## Others

### Qualitative Results
  * The figure below shows line-of-sight magnetograms from September 3th, 2017 23:00 UT to September 5th, 2017, 23:00 UT. An X-class solar flare occurred at 12:02 on September 6, 2017, and the model was able to predict the correct maximum solar flare class.
  
    ![magnetogram_256](https://user-images.githubusercontent.com/75234574/148938052-5d2a017e-c8fd-4f4f-9c10-0226e447c939.gif)


### Developer's env.
Hardware
- Ubuntu 20.04LTS
- GPU: GeForce RTX 3080 laptop
- GPU RAM: 16GB GDDR6
- CPU: Core i9 11980HK
- RAM: 64 GB
Libraries
- CUDA 11.1



