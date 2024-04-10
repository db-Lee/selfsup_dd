# Self-Supervised Dataset Distillation for Transfer Learning
This is the official Pytorch implementation for the paper ["**Self-Supervised Dataset Distillation for Transfer Learning**", in ICLR 2024.](https://openreview.net/forum?id=h57gkDO2Yg)

## Summary
<img align="middle" width="1000" src="https://github.com/db-Lee/selfsup_dd/blob/main/assets/concept.png">
Dataset distillation aims to optimize a small set so that a model trained on the set achieves performance similar to that of a model trained on the full dataset. While many supervised methods have achieved remarkable success in distilling a large dataset into a small set of representative samples, however, they are not designed to produce a distilled dataset that can be effectively used to facilitate self-supervised pre-training. To this end, we propose a novel problem of distilling an unlabeled dataset into a set of small synthetic samples for efficient self-supervised learning (SSL). We first prove that a gradient of synthetic samples with respect to a SSL objective in naive bilevel optimization is biased due to the randomness originating from data augmentations or masking for inner optimization. To address this issue, we propose to minimize the mean squared error (MSE) between a model's representations of the synthetic examples and their corresponding learnable target feature representations for the inner objective, which does not introduce any randomness. Our primary motivation is that the model obtained by the proposed inner optimization can mimic the self-supervised target model. To achieve this, we also introduce the MSE between representations of the inner model and the self-supervised target model on the original full dataset for outer optimization. We empirically validate the effectiveness of our method on transfer learning.

&nbsp;

__Contribution of this work__
- We propose a new problem of self-supervised dataset distillation for transfer learning, where we distill an unlabeled dataset into a small set,  
    pre-train a model on it, and fine-tune it on target tasks. 
- We have observed training instability when utilizing existing SSL objectives in bilevel optimization for self-supervised dataset distillation. Furthermore, we prove that a gradient of the SSL objectives with data augmentations or masking inputs is a biased estimator of the true gradient.
- To address the instability, we propose KRR-ST using MSE without any randomness at an inner loop. For the inner loop, we minimize MSE between a model representation of synthetic samples and target representations. For an outer loop, we minimize MSE between the original data representation of the model from inner loop and that of the model pre-trained on the original dataset. 
- We extensively validate our proposed method on numerous target datasets and architectures, and show that ours outperforms supervised dataset distillation methods.

## Dependencies
This code is written in Python. Dependencies include
* python >= 3.10
* pytorch = 2.1.2
* torchvision = 0.16.2
* tqdm
* korina = 0.7.1
* transformers = 4.36.2

```bash
conda env create -f environment.yml
conda activate dd
```

## Data and Model Checkpoints
* Download **Full Data**(~40GB) from [here](https://drive.google.com/file/d/1P0zwURUbVsqoVgIRcIZXGAtGrkRvGvH0/view?usp=sharing). 
* Download **Distilled Data**(~702MB) from [here](https://drive.google.com/file/d/1vDghSAUnmdWdGJgx9iK8dMOwoId0nuKF/view?usp=sharing).
* Download **Target (Teacher) Model Checkpoints**(~158MB) from [here](https://drive.google.com/file/d/1IuN4rhlB5UuJX_jrbVIWEBXo10QWHPBE/view?usp=sharing).

directory should be look like this:
```shell
┌── datasets/
  ┌── aircraft/
    ┌── X_te_32.pth
    ├── ...
    └── Y_tr_224.pth
  ├── cars/
      ...
  └── tinyimagenet/
  
├── synthetic_data/
  ┌── cifar100/
    ┌── dm/
        ┌── x_syn.pt
        └── y_syn.pt
    ├── ...
    └── random/
  ├── ...
  └── tinyimagenet/

└── teacher_ckpt/
  ┌── barlow_twins_resnet18_cifar100.pt
  ├── ...
  └── teacher_cifar10.pt
```

## Dataset Distillation
To distill **CIFAR100**, run the following code:
```bash
python train.py --exp_name EXP_NAME (e.g. "cifar100_exp") --data_name cifar100 --outer_lr 1e-4 --gpu_id N
```

To distill **TinyImageNet**, run the following code:
```bash
python train.py --exp_name EXP_NAME (e.g. "tinyimagenet_exp") --data_name tinyimagenet --outer_lr 1e-5 --gpu_id N
```

To distill **ImageNet 64x64**, run the following code:
```bash
python train.py --exp_name EXP_NAME (e.g. "imagenet_exp") --data_name imagenet --outer_lr 1e-5 --gpu_id N
```

## Transfer Learning
To reproduce **transfer learning with CIFAR100 (Table 1)**, run the following code:
```bash
python test_scratch.py --source_data_name cifar100 --target_data_name full --gpu_id N
python test.py --source_data_name cifar100 --target_data_name full --method METHOD (["random", "kmeans", "dsa", "dm", "mtt", "kip", "frepo", "krr_st"]) --test_model base --gpu_id N
```

To reproduce **transfer learning with TinyImageNet (Table 2)**, run the following code:
```bash
python test_scratch.py --source_data_name tinyimagenet --target_data_name full --gpu_id N
python test.py --source_data_name tinyimagenet --target_data_name full --method METHOD (["random", "kmeans", "dsa", "dm", "mtt", "frepo", "krr_st"]) --test_model base --gpu_id N
```

To reproduce **transfer learning with ImageNet 64x64 (Table 3)**, run the following code:
```bash
python test_scratch.py --source_data_name imagenet --target_data_name full --gpu_id N
python test.py --source_data_name imagenet --target_data_name full --method METHOD (["random", "frepo", "krr_st"]) --test_model base --gpu_id N
```

To reproduce **architecture generalization with TinyImageNet (Figure 3)**, run the following code:
```bash
python test_scratch.py --source_data_name tinyimagenet --target_data_name aircraft_cars_cub2011_dogs_flowers --test_model ARCHITECTURE (["vgg", "alexnet", "mobilenet", "resnet10"]) --gpu_id N
python test.py --source_data_name tinyimagenet --target_data_name aircraft_cars_cub2011_dogs_flowers --method METHOD (["random", "kmeans", "dsa", "dm", "mtt", "frepo", "krr_st"]) --test_model ARCHITECTURE (["vgg", "alexnet", "mobilenet", "resnet10"]) --gpu_id N
```

To reproduce **target data-free knowledge distillation with TinyImageNet (Table 4)**, run the following code:
```bash
python test_kd.py --source_data_name tinyimagenet --method METHOD (["gaussian", "random", "kmeans", "dsa", "dm", "mtt", "frepo", "krr_st"]) --test_model ARCHITECTURE (["base", "vgg", "alexnet", "mobilenet", "resnet10"]) --gpu_id N
```

## Reference
To cite our paper, please use this BibTex
```bibtex
@inproceedings{lee2024selfsupdd,
  title={Self-Supervised Dataset Distillation for Transfer Learning},
  author={Dong Bok Lee and and Seanie Lee and Joonho Ko and Kenji Kawaguch and Juho Lee and Sung Ju Hwang},
  booktitle={Proceedings of the 12th International Conference on Learning Representations},
  year={2024}
}
```
