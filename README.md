# Monte Carlo Bulls

随缘炼丹，随缘赚钱

## Runnable training demo

```bash
codna activate stock-venv
python training_demo.py
```

## 模型存放位置
learner参数`model_save_prefix`确定，默认为`/mnt/data3/rl-data/training_res`之下的某一目录

## 调参/实验管理

### 启动前准备
- 修改实验参数，主要是调试什么参数。参考[nni_profile/test.yaml](nni_profile/test.yaml)和[nni文档](https://nni.readthedocs.io/en/latest/hpo/search_space.html)。
    - 其中参数以`<module>$<param name>`的形式给出，如`learner_config$batch_size`，意为`learner_config`模块下的`batch_size`参数。
      对应模块的参数将会代理至类构造函数或config类中。
    （过程参考[该文件中`get_param_from_nni`](training/util/exp_management.py)）
    - 默认参数在[nni_profile/default.yaml](nni_profile/default.yaml)中给出，缺省将使用这些参数。
- 可以考虑新建一个.yaml文件

### 启动实验
```bash
nnictl create --config nni_profile/test.yaml --port 18080
```

打开[http://192.168.194.52:18080](http://192.168.194.52:18080)，中间结果和最终结果看图说话。

实验将会把模型保存到`/mnt/data3/rl-data/training_res/<exp_id>/<trial_id>`之下。

