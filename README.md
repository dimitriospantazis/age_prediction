# Age Prediction through MEG Analysis using FHNN 
Age Prediction using [Fully Hyperbolic Neural Networks](https://arxiv.org/abs/2105.14686) 

```
@article{chen2021fully,
  title={Fully Hyperbolic Neural Networks},
  author={Chen, Weize and Han, Xu and Lin, Yankai and Zhao, Hexu and Liu, Zhiyuan and Li, Peng and Sun, Maosong and Zhou, Jie},
  journal={arXiv preprint arXiv:2105.14686},
  year={2021}
}
{"mode":"full","isActive":false}
```

# Codes for Network Embedding
The codes are based on [HGCN](https://github.com/HazyResearch/hgcn) repo. Codes related to our HyboNet are remarked below.

```
📦gcn
 ┣ 📂data
 ┣ 📂layers
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜att_layers.py
 ┃ ┣ 📜hyp_layers.py    # Defines Lorentz Graph Convolutional Layer
 ┃ ┗ 📜layers.py
 ┣ 📂manifolds
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base.py
 ┃ ┣ 📜euclidean.py
 ┃ ┣ 📜hyperboloid.py
 ┃ ┣ 📜lmath.py         # Math related to our manifold
 ┃ ┣ 📜lorentz.py       # Our manifold
 ┃ ┣ 📜poincare.py
 ┃ ┗ 📜utils.py
 ┣ 📂models
 ┃ ┣ 📜__init__.py
 ┃ ┣ 📜base_models.py
 ┃ ┣ 📜decoders.py      # Include our HyboNet decoder
 ┃ ┗ 📜encoders.py      # Include our HyboNet encoder
 ┣ 📂optim
 ┣ 📂utils
 ```

## 1. Usage
Arguments passed to program:

`--task` Specifies the task. Can be [lp, nc], lp denotes link prediction, and nc denotes node classification.

`--dataset` Specifies the dataset. Can be [airport, disease, cora, pubmed].

`--lr` Specifies the learning rate.

`--dim` Specifies the dimension of the embeddings.

`--num-layers` Specifies the number of the layers.

`--bias` To enable the bias, set it to 1.

`--dropout` Specifies the dropout rate.

`--weight-decay` Specifies the weight decay value.

`--log-freq` Interval for logging.

For other arguments, see `config.py`

Example Run:

! python train_graph_iteration.py \
    --task lp \
    --act None \
    --dataset cam_can_multiple\
    --model HyboNet \
    --lr 0.05 \
    --dim 3 \
    --num-layers 2 \
    --bias 1 \
    --dropout 0.25 \
    --weight-decay 1e-3 \
    --manifold Lorentz \
    --log-freq 5 \
    --cuda -1 \
    --patience 500 \
    --grad-clip 0.1 \
    --seed 1234 \
    --save 1
