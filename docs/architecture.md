
## 框架讲解

![image-20230331223801343](https://s2.loli.net/2023/04/03/kOstofQECU7nIS9.png)

以上是预训练框架，先用self-attension将有的三维数据映射二维，然后将所有数据拼在一起再做一个cross attension把所有数据再压成二维，加一个position的embedding之后过一个基于预测时间的mlp直接喂进Transformer里再进入Predict层算所有的输入变量的预测值，最后根据预测的时间算一个loss，流程简单粗暴，但形成范式。

具体而言，模型随机生成𝑡 ∼ 𝒰[6, 168]的预测时间，然后根据此变量计算结果和Loss。

#### Loss

![image-20230331224728678](https://s2.loli.net/2023/04/03/6IleNXoqsCpG7QW.png)

Loss函数用了一个纬度加权均方误差计算方法，这里看公式就能看出来，一目了然的加权平均。


### Finetuning

![image-20230331225523319](https://s2.loli.net/2023/03/31/KtcwpmMZu9OA1XL.png)

对于用以上方法预训练的模型，微调很简单，分两种情况，第一种情况是预训练集包括了此要素就直接在训练集上跑，第二种情况就换掉Header和Prediction然后看情况要不要freeze掉预训练的参数跑，还可以再加一个neck来适配history。


