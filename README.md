## Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction
This repository contains the code for unsupervised group estimation applied to the trajectory prediction models.

**[Learning Pedestrian Group Representations for Multi-modal Trajectory Prediction](https://inhwanbae.github.io/publication/gpgraph/)**
<br>
<a href="https://InhwanBae.github.io/">Inhwan Bae</a>,
Jin-Hwi Park, and
<a href="https://scholar.google.com/citations?user=Ei00xroAAAAJ">Hae-Gon Jeon</a>
<br>Accepted to 
<a href="https://eccv2022.ecva.net/">ECCV 2022</a>


## GP-Graph Architecture
* Learns to assign each pedestrian into the most likely behavior group in unsupervised manner.
* Pedestrian group hierarchy graph
* Group-level latent vector sampling strategy


## Model Training
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