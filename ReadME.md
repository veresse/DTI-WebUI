# 药物-靶标相互作用预测平台
该平台依托深度学习模型的强大功能，并结合丰富的生物信息学大数据资源，可在药物设计初期便实现对药物与靶标相互作用关系的高精度预测，大幅提升了新药研发的速度与成功率。鉴于药物发现过程中复杂的动态变化和紧迫的时间需求，该平台运用深度学习算法细致模拟药物与靶标间的三维结构关系以及多种化学物理作用模式，旨在通过前端的精密筛选机制，大幅度减少药物研发周期，避免无效候选化合物不必要的后续试验投入，从而达到高效利用科研资源的目标。
# 依赖包
+ Python 3.8 
+ torch==2.1
+ gradio==4.28.3 
+ matplotlib 
+ numpy 
+ Pillow
+ Pillow
+ rdkit 
+ Requests
# 结构
+ README.md：此文件。
+ config：存放平台使用模型文件的路径
+ data_pre.py：数据处理。
+ gat.py：图神经网络。
+ hyperparameter.py：超参数
+ model1.py：DPAG 模型结构。
+ model2.py：MNDT 模型结构。
+ transformer.py：transformer网络
+ sw_tf.py：swin-transformer网络
+ interformer.py：interformer网络
# 运行
~~~
Python UI.py
~~~
