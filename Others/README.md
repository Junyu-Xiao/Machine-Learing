# 工具介绍
>***这个文件夹包含了一些小工具，如数据结构算法，评分算法，不定时更新。***

---

* ## KS值
### 算法原理
>KS值是$TPR$曲线与$FPR$曲线之间差值绝对值的最大值，代表着正负样本的区分度大小。常规的KS值计算需要遍历样本，时间复杂度为$O(n)\times O(m)$，其中$O(n)$为遍历的时间复杂度，$O(m)$为每步计算的复杂度，当样本量巨大时速度非常慢，为此对以上算法做出改进。
>
>新的算法代码的步骤如下：
>
>>１．对分数进行总排序。
>>２．对分数进行类内排序。
>>３．使用numpy.vectorize进行并行运算，每步的计算仅涉及到总排序、类内排序以及两个排序差。
>>４．计算差值绝对值的最大值。
>
>计算千万级数据的时间在10s内。

### 示例代码
>```python
>##调用脚本
>import sys
>sys.path.append('./Module')
>import KS_value as ks
>
>##计算ks值
>ks.KS_value(label, predict, label_kind = 1)
>```
>其中label是真实标签，predict是预测值，label_kind是需要计算ks值的标签名。

---

## To be continued

---

<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-23

