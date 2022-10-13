## Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction
This repository contains the code for unsupervised group estimation applied to the trajectory prediction models.

**[Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction](https://inhwanbae.github.io/publication/gpgraph/)**
<br>
<a href="https://InhwanBae.github.io/">Inhwan Bae</a>,
Jin-Hwi Park, and
<a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ">Hae-Gon Jeon</a>
<br>Accepted to 
<a href="https://eccv2022.ecva.net/">ECCV 2022</a>

<div align='center'>
  <img src="img/gpgraph-teaser-animated.webp" width=45%'>
  <img src="img/gpgraph-hierarchy-animated.webp" width=45%'>
</div>


## GP-Graph Architecture
* Learns to assign each pedestrian into the most likely behavior group in an unsupervised manner.
* Pedestrian group pooling&unpooling and group hierarchy graph for group behavior modeling.
* Group-level latent vector sampling strategy to share the latent vector between group members.


## Model Training
### Setup
**Environment**
<br>All models were trained and tested on Ubuntu 20.04 with Python 3.7 and PyTorch 1.9.0 with CUDA 11.1.

**Dataset**
<br>Preprocessed [ETH](https://web.archive.org/web/20190715200622/https://vision.ee.ethz.ch/datasets_extra/ewap_dataset_full.tgz) and [UCY](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data) datasets are included in this repository, under `./dataset/`. 
The train/validation/test splits are the same as those fond in [Social-GAN](https://github.com/agrimgupta92/sgan).

**Baseline models**
<br>This repository supports the [SGCN](https://arxiv.org/abs/2104.01528) baseline trajectory predictor.
We have included model source codes from [their official GitHub](https://github.com/shuaishiliu/SGCN/tree/0ff25cedc04852803787196e83c0bb941d724fc2) in `model_baseline.py` 

### Train GP-Graph
To train our GPGraph-SGCN on the ETH and UCY datasets at once, we provide a bash script `train.sh` for a simplified execution.
```bash
./train.sh
```
We provide additional arguments for experiments: 
```bash
./train.sh -t <experiment_tag> -d <space_seperated_dataset_string> -i <space_seperated_gpu_id_string>

# Examples
./train.sh -d "hotel" -i "1"
./train.sh -t onescene -d "hotel" -i "1"
./train.sh -t allinonegpu -d "eth hotel univ zara1 zara2" -i "0 0 0 0 0"
```
If you want to train the model with custom hyper-parameters, use `train.py` instead of the script file.


## Model Evaluation
### Pretrained Models
We have included pretrained models in the `./checkpoints/` folder.

### Evaluate GP-Graph
You can use `test.py` to evaluate our GPGraph-SGCN model. 
```bash
python test.py
```


## Citation
If you find this code useful for your research, please cite our paper :)

```bibtex
@inproceedings{bae2022gpgraph,
  title={Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction},
  author={Bae, Inhwan and Park, Jin-Hwi and Jeon, Hae-Gon},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2022}
}
```

### Acknowledgement
Part of our code is borrowed from [SGCN](https://github.com/shuaishiliu/SGCN/tree/0ff25cedc04852803787196e83c0bb941d724fc2). 
We thank the authors for releasing their code and models.