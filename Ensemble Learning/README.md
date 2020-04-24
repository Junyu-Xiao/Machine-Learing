# 集成学习介绍

>***主要介绍一些集成学习的方法。***

---

* ## Boosting
>Boosting是将弱学习器以串联的方式结合到一起的集成方式，在弱学习器低方差高偏差的特性下，逐步降低模型的偏差，主要的Boosting模型有adaboost、xgboost、GBDT等。详情请看Boosting文件夹。

---

* ## Bagging
>Bagging是将强学习器以并联的方式结合到一起的集成方式，在强学习器高方差低偏差的特性下，增加基学习器的数量，以降低模型的方差，主要的Bagging模型有RandomForest。详情请看Bagging文件夹。

---

* ## Stacking
>Stacking是将多个初级模型的输出作为次级模型的输入的一种模型融合方式。为了保证不同层模型之间的信息不发生泄露，因此采用多折交叉预测的方式。详情请看模型融合文件夹。

---

* ## Blending
>Blending与Stacking类似，主要的区别在于防止信息泄露的方式为将数据集划分为多层数据，不同层之间的数据相互独立。详情请看模型融合文件夹。

---

<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-20