## 代码结构

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


