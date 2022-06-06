# On the performance of deep generative models of realistic sat instances

This repository collects the implementation of the four models published in "On the performance of deep generative models of realistic sat instances" [1]. All these models follow the generation methodology proposed in "G2SAT: learning to generate SAT formulas" [2]. The original G2SAT code is available in [https://github.com/JiaxuanYou/G2SAT](https://github.com/JiaxuanYou/G2SAT).

## Installation

- Install PyTorch (tested on 1.0.0), please refer to the offical website for further details
```bash
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
- Install PyTorch Geometric (tested on 1.1.2), please refer to the offical website for further details
```bash
pip install --verbose --no-cache-dir torch-scatter
pip install --verbose --no-cache-dir torch-sparse
pip install --verbose --no-cache-dir torch-cluster
pip install --verbose --no-cache-dir torch-spline-conv (optional)
pip install torch-geometric
```
- Install networkx (tested on 2.3), make sure you are not using networkx 1.x version!
```bash
pip install networkx
```
- Install tensorboardx
```bash
pip install tensorboardX
```

# References

[1] Iván Garzón, Pablo Mesejo, and Jesús Giráldez-Cru. On the performance of deep generative models of realistic sat instances. In Proc. of the 25th Int. Conf. on Theory and Applications of Satisfiability Testing (SAT 2022).

[2] Jiaxuan You, Haoze Wu, Clark W. Barrett, Raghuram Ramanujan, and Jure Leskovec. G2SAT: learning to generate SAT formulas. In Proc. of the Annual Conference on Neural Information Processing Systems (NeurIPS 2019), pages 10552–10563, 2019.
