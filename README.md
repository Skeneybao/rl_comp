# Monte Carlo Bulls

## 环境
```bash
conda create -f environment.yml
```


## 数据处理


### 降采样

脚本: [data_processing/nearest_time.py](data_processing/nearest_time.py)

- 需要修改变量`training_set`和`output_path_base`，指定输入的原始数据位置，以及输出的数据位置。

### 数据集划分

- 训练集包括直至20200219（包含）
- 验证集包括20200220, 20200221, 20200224

### 修改代码中的路径定义

全局搜索`/mnt/data3/rl-data`，修改下面文件中的对应路径至实际路径
- [training/default_param.py](training/default_param.py)
- [training/util/exp_management.py](training/util/exp_management.py)
- [training/env/trainingEnv.py](training/env/trainingEnv.py)
- [run_training_nni.py](run_training_nni.py)

## 实验

在项目根目录执行
```bash
nnictl create --config nni_profile/reproduce_1.yaml --port <可用端口> 
```

实验配置有两个
- reproduce_1.yaml：相对稳定
- reproduce_2.yaml：不稳定。同时由于训练的每一个环节都具备随机性且难以消除（模型参数初始化，epsilon-greedy选择，训练采样，cudnn行为等），期待运行十余个模型，可以出现一个稍微可用的。

可以修改`trial_concurrency`控制并发度