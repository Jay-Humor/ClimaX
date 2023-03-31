<p align="center">
  <img src="https://user-images.githubusercontent.com/1785175/215624212-fc92ccb1-f14c-4cb6-982f-61f50b9f3c21.png" width="320px">
</p>

[![Documentation](https://img.shields.io/badge/docs-passing-brightgreen)](https://microsoft.github.io/ClimaX)
[![Paper](https://img.shields.io/badge/arXiv-2301.10343-blue)](https://arxiv.org/abs/2301.10343)

This repository contains code accompanying the paper [**ClimaX: A foundation model for weather and climate**](https://arxiv.org/abs/2301.10343).

For details about usage please see [documentation](https://microsoft.github.io/ClimaX).
If you have any questions or suggestions please open a [discussion](https://github.com/microsoft/ClimaX/discussions). If you notice a bug, please open an [issue](https://github.com/microsoft/ClimaX/issues).

中科曙光 ClimaX文档

## 框架讲解

![image-20230331223801343](https://s2.loli.net/2023/03/31/ayhplTVDx3quPC6.png)

以上是预训练框架，先用self-attension将有的三维数据映射二维，然后将所有数据拼在一起再做一个cross attension把所有数据再压成二维，加一个position的embedding之后过一个基于预测时间的mlp直接喂进Transformer里再进入Predict层算所有的输入变量的预测值，最后根据预测的时间算一个loss，流程简单粗暴，但形成范式。

具体而言，模型随机生成𝑡 ∼ 𝒰[6, 168]的预测时间，然后根据此变量计算结果和Loss。

#### Loss

![image-20230331224728678](https://s2.loli.net/2023/03/31/nklASEGC7MOsfoP.png)

Loss函数用了一个纬度加权均方误差计算方法，这里看公式就能看出来，一目了然的加权平均。


### Finetuning

![image-20230331225523319](https://s2.loli.net/2023/03/31/KtcwpmMZu9OA1XL.png)

对于用以上方法预训练的模型，微调很简单，分两种情况，第一种情况是预训练集包括了此要素就直接在训练集上跑，第二种情况就换掉Header和Prediction然后看情况要不要freeze掉预训练的参数跑，还可以再加一个neck来适配history。

### Discussion

改进的地方感觉有不少，第一个是觉得一张图或许不能体现一个区域的数据，第二个是认为可以加一个pretrain的stage，来重建mask掉的数据，毕竟，在预测之前先了解一些基本的气候知识是一件好事，不是吗？

## 代码

```
climax
    global_forecast # 全球预报
        __init__.py
        datamodule.py
        module.py
        train.py
    pretrain # 预训练
        __init__.py
        datamodule.py
        module.py
        train.py
        dataset.py
    regional_forecast # 区域预报
        __init__.py
        datamodule.py
        module.py
        train.py
        arch.py
    utils # 一些函数
        data_utils.py
        lr_scheduler.py
        metrics.py
        pos_embed.py
    arch.py # 总体框架
    __init__.py
	
```



未完待续。。。
