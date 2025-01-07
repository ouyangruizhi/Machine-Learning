# Machine-Learning
北京林业大学统计学zx老师**统计学习导论**课程作业
## 线性回归模型
- 第一次作业  
- 主题：房价预测    
- 线性回归模型：LASSO回归、Ridge岭回归、ElasticNet弹性网、Lars最小角度模型、LARS-Lasso最小角拉索和SGD随机梯度下降
- 主要就是运用一下模型
## Logistic回归
- 第二次作业
- 数据来源：sklearn库自带的手写数字、鸢尾花数据集
- Logistic回归实现二分类与多分类
## 决策树Decision Tree
- 第三次作业
- 数据来源：阿里天池提供的[Banking Dataset Classification银行数据集分类](https://tianchi.aliyun.com/dataset/92775)
- 决策树建模与模型评估
- 决策树模型稳健性实验
- 决策树与Logistic的比较 
## 建模流程
- 第四次作业
- 数据来源：阿里天池提供的[Banking Dataset Classification银行数据集分类](https://tianchi.aliyun.com/dataset/92775)
- 内容：银行定期存款营销——采用更为 __激进__ 的策略
- 建模流程：业务理解、数据探索、数据预处理、不平稳数据处理、模型评估与选择
- 模型：Logistic回归、Decision Tree决策树、SVM分类和KNN
- 模型评估：Accuracy、precision、Recall、F1、混淆矩阵、Lift曲线
## 信用评分
- 第五次作业
- 数据来源：阿里云天池提供的[Give Me Some Credit给我一些荣誉](https://tianchi.aliyun.com/dataset/89334)
- 内容：信用评分——采用更为 __保守__ 的策略
- 模型：Logistic回归、Decision Tree决策树、SVM分类和KNN
- 模型评估：Accuracy、precision、Recall、F1、混淆矩阵、ROC曲线&AUC值
## Decision Tree的辅助使用
- 第六次作业
- 数据来源：阿里云天池提供的[Give Me Some Credit给我一些荣誉](https://tianchi.aliyun.com/dataset/89334)
- 决策树分组处理缺失值
- 决策树实现特征分箱
- 决策树实现交互作用检测
- 决策树实现分层建模
## 聚类分析
- 第七次作业
- 数据来源：Kaggle提供的[Online Retail Data Set from UCI ML repo](https://www.kaggle.com/datasets/jihyeseo/online-retail-data-set-from-uci-ml-repo)
- 内容：RBF智能营销
- 模型：Hierarchical clustering层次聚类、K-means、MiniBatchKMeans、Affinity Propagation、Spectral clustering、DBSCAN、
## 推荐算法之关联分析
- 第八次作业
- 数据来源：GitHub还是知乎找的数据，我给忘了，需要请Email我
- 内容：关联分析(关联规则、购物篮分析)
- 模型：Apriori算法、FP-growth算法
## 推荐算法之协同过滤
- 第九次作业
- 数据来源：Grouplens提供的数据集[MovieLens](https://grouplens.org/datasets/movielens/1m/)
- 调参关键在于算法、K值、相似度的选择
- 基于用户的协同过滤User-based CF
- 基于物品的协同过滤Item-based CF：KNNBasic, KNNWithMeans, KNNWithZScore, KNNBaseline
## 神经网络的无监督学习
- 第十次作业
- 数据来源：sklearn库直接导入手写数字数据
- sklearn提供的算法：受限玻尔兹曼机（RBM）
- MiniSom库提供的SOM模型
## RBF径向基网络
- 第十一次作业
- 数据来源：阿里天池提供的[风电功率预测](https://tianchi.aliyun.com/dataset/159885)
- 模型：基于RBF的神经网络
- 依赖库：keras
- 重点：神经网络调参
## MLP多层感知器
- 第十二次作业
- 数据来源：阿里天池提供的[风电功率预测](https://tianchi.aliyun.com/dataset/159885)
- 模型：基于MLP的神经网络
- 依赖库：keras
- 重点：神经网络调参
## CNN卷积神经网络
- 第十三次作业
- 数据来源：keras自带的MNIST手写数字识别
- 模型：CNN
## LSTM、GRU
- 第十四次作业
- 数据来源：akshare的碳价数据
- 模型：CNN、LSTM、GRU、Attention
## 集成学习之Boosting
- 第十五次作业
- 数据来源：阿里天池提供的[Banking Dataset Classification银行数据集分类](https://tianchi.aliyun.com/dataset/92775)
- 模型：Adaboost-Logistic、XGBoost、LightGBM
- 重点在于 __调参__
## 集成学习之Bagging
- 第十六次作业
- 数据来源：阿里天池提供的[Banking Dataset Classification银行数据集分类](https://tianchi.aliyun.com/dataset/92775)
- 模型：Bagging组合Decision Tree/SVM/Logistic、Random Forestry
## 集成学习之Stacking&Voting
- 第十七作业
- 数据来源：阿里天池提供的[Banking Dataset Classification银行数据集分类](https://tianchi.aliyun.com/dataset/92775)
- 模型：Stacking组合Random Forestry/SVM/HGB、Voting组合Random Forestry/SVM/HGB
## 生成模型
- 第十八次作业
- 数据来源：pytorch自带的MNIST手写数字识别
- 模型：自编码器AE、变分自编码器VAE、生成对抗网络GAN、GAN变体：深度卷积生成对抗网络DCGAN、Flow-based Model：RealNVP、扩散模型Diffusion Model、隐扩散模型Latent Diffusion Model
- 注：有点难，建议B站看李宏毅老师的视频，反正我是没学懂
## 流行学习Manifold Learning
- 第十九次作业
- 数据来源：keras自带的MNIST手写数字识别
- 模型：PCA、多维缩放MDS、等距映射Isomap、局部线性嵌入LLE、LLE的改进MLLE、HLLE、LTSA、t分布随机邻域嵌入t-SNE、频谱嵌入Spectral Embedding
- 依赖库：Sklearn
## 推荐算法之点击率预测CTR
- 第二十次作业
- 数据来源：kaggle 2015年竞赛[Click-Through Rate Prediction](https://www.kaggle.com/competitions/avazu-ctr-prediction)
***
__主要依赖库：[sklearn](https://scikit-learn.org/stable/)、[keras](https://keras-cn.readthedocs.io/en/latest/)、[tensorflow](https://tensorflow.google.cn/?hl=zh-cn)、[pytorch](https://pytorch-cn.readthedocs.io/zh/latest/)、[scipy](https://docs.scipy.org/doc/scipy-1.13.0/index.html)、[imblearn](https://imbalanced-learn.org/stable/index.html)、[surprise](https://surprise.readthedocs.io/en/stable/index.html)、[mlxtend](https://rasbt.github.io/mlxtend/)__  
__主要数据来源：[天池](https://tianchi.aliyun.com/dataset?spm=a2c22.27124976.J_3941670930.20.71de132aJGzOYY)、[kaggle](https://www.kaggle.com)、[sklearn库自带数据集](https://scikit-learn.org/stable/api/sklearn.datasets.html)、[keras自带数据集](https://keras-cn.readthedocs.io/en/latest/legacy/other/datasets/)、[pytorch自带数据集](https://pytorch-cn.readthedocs.io/zh/latest/torchvision/torchvision-datasets/)__   
在算力上有困难，可以尝试使用Kaggle免费的GPU和TPU  
如有任何问题，请Email我：ouyangruizhi@bjfu.edu.cn
