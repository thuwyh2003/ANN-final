## **11-25   尝试使用pytorch跑通源代码**

大致参照源代码提供的README操作，我在操作过程中遇到了一些bug，目前已解决，主要有以下几个，后续随时补充：

#### 虚拟环境     transfg2

python 3.12    torch==2.5.1   torchvision==0.20.1

新创建的虚拟环境还有一些需要安装的包，我已经添加在requirements.txt

#### 需要下载apex包，

建议git clone [https://github.com/NVIDIA/apex](https://github.com/NVIDIA/apex)

cd apex
python setup.py install   安装apex 0.1

#### dataset.py需要修改数据读取

scipy.misc.imread()版本过低，需要换成pillow库。我已经修改过，可直接pull下来

#### 运行训练代码

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1  --nproc_per_node=4   train.py --dataset CUB_200_2011 --split overlap --num_steps 10000 --fp16 --name sample_run --pretrained_dir /home/wyh/ANN/TransFG/imagenet21k_ViT-B_16.npz      分布式训练代码我已修改，可直接pull下来

## 11.29

汪钰涵  修改TransFG的models/modeling.py的代码

TODO:

1. [X]  MLP Block
2. [X]  Encoder Block
3. [X]  Vision Transformer
4. [X]  Patch embedding
5. [X]  Multi-head Attention
6. [X]  PSM（part attention）

持续更新memo，这是我在操作的时候总结的一些问题，hopefully，能给大家提供帮助

## 12.15

崔灏睿 修改部分bug
分布式训练代码api存在问题

```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1  --nproc_per_node=4   train_jt.py --dataset CUB_200_2011 --split overlap --num_steps 10000 --fp16 --name sample_run --pretrained_dir pretrained_dir/imagenet21k_ViT-B_32.npz
```

12.17

汪钰涵

python3 train_jt.py --dataset CUB_200_2011 --split overlap --num_steps 10000 --fp16 --name sample_run --pretrained_dir /home/wyh/ANN/TransFG/imagenet21k_ViT-B_16.npz
<<<<<<< HEAD

CUDA_VISIBLE_DEVICES=0 python3 train_jt.py --dataset dog --split overlap --num_steps 10000 --fp16 --name sample_run --pretrained_dir ./pretrained_dir/imagenet21k_ViT-B_16.npz
=======
12.22
杨雨晴
CUDA_VISIBLE_DEVICES=0 python3 train_jt.py --dataset CUB_200_2011 --split overlap --num_steps 10000 --fp16 --name sample_run --pretrained_dir ../pretrained_dir/imagenet21k_ViT-B_16.npz
torch 非分布式训练
torchrun train.py --dataset CUB_200_2011 --split overlap --num_steps 10000 --fp16 --name sample_run --pretrained_dir ../pretrained_dir/imagenet21k_ViT-B_16.npz
>>>>>>> 8666366cc9ff103996cd0ac4b923ed51bc4b4b49
