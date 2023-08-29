# [ICML23] Lowering the Pre-training Tax for Gradient-based Subset Training: A Lightweight Distributed Pre-Training Toolkit

### Introduction
Coreset is a promising approach that reduces computation and memory cost during training of large models. 
Most recent algorithms tend to use error or loss during the training as their metric for valuation.
However, to infer a reliable importance value of each data item, gradients need to be stabilized before the valuation 
that often cannot be met for the early-stage gradients from severely under-trained weights. 
This work primarily focus on reducing the *pre-training* or warm-up stage for gradient-based subset selection algorithms. 

For details about the method, please check our [paper](https://proceedings.mlr.press/v202/ro23a.html).
The code adopted and implemented on top of two codebases: [Deepcore](https://github.com/PatrickZH/DeepCore) and [Random Pruning (ICLR'22)](https://github.com/VITA-Group/Random_Pruning). 

### Citation
Please cite the following paper if you use LiPT. 
```
@InProceedings{pmlr-v202-ro23a,
  title = 	 {Lowering the Pre-training Tax for Gradient-based Subset Training: A Lightweight Distributed Pre-Training Toolkit},
  author =       {Ro, Yeonju and Wang, Zhangyang and Chidambaram, Vijay and Akella, Aditya},
  booktitle = 	 {Proceedings of the 40th International Conference on Machine Learning},
  pages = 	 {29130--29142},
  year = 	 {2023},
  editor = 	 {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = 	 {202},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {23--29 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v202/ro23a/ro23a.pdf},
  url = 	 {https://proceedings.mlr.press/v202/ro23a.html},
}

```

### Datasets
We included implementation of CIFAR10, CIFAR100, and ImageNet.

### Models
We included implementation of ResNet and WideResNet. 

### Subset Selection Algorithms
We included implementation of Glister, GraNd, and GradMatch. 

### Pretraining Parameters
TO ADD

### Example
Selecting with Glister and training on the coreset with fraction 0.1.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Glister --model ResNet18 --lr 0.1 -sp ./result --batch 128
```

### References
1. Killamsetty, K., Durga, S., Ramakrishnan, G., De, A., Iyer, R.: Grad-match: Gradient matching based data subset selection for efficient deep model training. In: ICML. pp. 5464–5474 (2021)
2. Killamsetty, K., Sivasubramanian, D., Ramakrishnan, G., Iyer, R.: Glister: Generalization based data subset selection for efficient and robust learning. In: Proceedings of the AAAI Conference on Artificial Intelligence (2021)
3. Paul, M., Ganguli, S., Dziugaite, G.K.: Deep learning on a data diet: Finding important examples early in training. arXiv preprint arXiv:2107.07075 (2021)
4. Guo, C., Zhao, B. and Bai, Y., 2022, July. Deepcore: A comprehensive library for coreset selection in deep learning. In Database and Expert Systems Applications: 33rd International Conference, DEXA 2022, Vienna, Austria, August 22–24, 2022, Proceedings, Part I (pp. 181-195). Cham: Springer International Publishing.
5. Liu, S., Chen, T., Chen, X., Shen, L., Mocanu, D.C., Wang, Z. and Pechenizkiy, M., 2022. The unreasonable effectiveness of random pruning: Return of the most naive baseline for sparse training. arXiv preprint arXiv:2202.02643.
