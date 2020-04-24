# Scatter使用教程

>***该脚本是基于Pyecharts所编写的，是已配置好样式的scatter脚本。***

---

## 适用方向
>因为气泡图的特性，所有数据点之间是独立非连续的，因此用于统计经办行为以及经办绩效等独立个体的数据时，可以由几何距离直观的反映出经办间的差距。除此之外，由于气泡图除了横纵坐标，还可以用气泡大小来展示第三个维度的信息，目前应用到的方向如下：
>
>### 经办绩效：
>
>横轴表示户数，纵轴表示金额，气泡大小表示奖牌数。
>### 经办行为：
>横轴表示拨打量，纵轴表示打标数，气泡大小表示催回数。

---

##　示例代码
### 1. 加载所需包

>```python
>##载入所需包
>import sys
>sys.path.append('/home/junyu/PycharmProjects/work_plain/F1_example')
>import achievement as achi
>import pandas as pd
>```
>其中achievement包就是配置好的scatter脚本。

### 2. 载入数据并转化数据

>```python
>##载入数据
>data = pd.read_csv('./PycharmProjects/work_plain/F1_example/data.csv')
>
>##转为字典
>data_dict = data.to_dict(orient='records')
>```
>data的格式如下：
>
>|   地区   |  阶段  | 处理人 | 还款户数 | ...  | 金额排行 |
>| :------: | :----: | :----: | :------: | :--: | :------: |
>| 徐州二部 | Y1大额 |  李岩  |  198.0   | ...  |    1     |
>| 徐州一部 | Y1大额 |  魏雯  |  196.5   | ...  |    2     |
>| 徐州二部 | Y1大额 | 王娟03 |  191.5   | ...  |    3     |
>|   ...    |  ...   |  ...   |   ...    | ...  |   ...    |
>| 徐州二部 | Y1大额 | 王媛媛 |   29.5   | ...  |    41    |
>
>data_dict是将每一行的数据转为字典格式，格式如下：
>```python
>[{'地区': 'xxx',
>  '阶段': 'xxx',
>  '处理人': 'xxx',
>  '还款户数': xxx,
>  '还款金额': xxx,
>  '日均户数': xxx,
>  '日均金额': xxx,
>  '阶段天数': xx,
>  '备注': 'xx ',
>  '金额排行': xx},...]
>```

### 3. 画出图表

>```python
>##画图
>scatter = achi.Achievement(data_dict)
>scatter.Data_Index(x_name = '还款户数', y_name = '还款金额', size_name = ['阶段天数'], size_coef= [1.1], series = '地区', name = '处理人')
>scatter.plot_axis(width = '1000px', height='600px', x_min=0, x_max=210, y_min=0, y_max=400000, theme = ThemeType.LIGHT)
>plot = scatter.Plot(jine=[235000], hushu=[143], coef=[1.0], date = '2020-03-25', title = '绩效成绩')
>
>##导出图表
>###在jupyter-notebook中显示
>plot.render_notebook()
>###保存为html文件
>plot.render('./plot.html')
>```
>scatter.Data_Index()是控制各个数据在图表中的展示，x_name是横坐标的数据名称；y_name是纵坐标的数据名称；size_name是控制气泡大小的数据列表（例如红钻、蓝钻由两个数据控制气泡大小）；size_coef是对应的size_name列表中数据的权值大小（例如红钻的权值是1.5，蓝钻的权值是1.1）；series是以什么数据将气泡归类；例如将经办按照地区归类；name是气泡的标签名称。
>
>scatter.plot_axis()是控制图表的大小坐标以及主题等内容，width/height表示图表的横/纵像素大小；x_min/x_max表示图表横坐标的起点与终点；y_min/y_max表示图表纵坐标的起点与终点；theme表示图表的颜色主题（内置的主题有：ThemeType.LIGHT、ThemeType.DARK、ThemeType.CHALK、ThemeType.ESSOS、ThemeType.INFOGRAPHIC、ThemeType.MACARONS、ThemeType.PURPLE_PASSION、ThemeType.ROMA、ThemeType.ROMANTIC、ThemeType.SHINE、ThemeType.VINTAGE、ThemeType.WALDEN、ThemeType.WESTEROS、ThemeType.WONDERLAND）。
>
>scatter.Plot()是控制达标线的展示，jine是金额达标线列表（[10000, 20000]表示有10000和20000两条达标线）；hushu是户数达标线列表；coef是户数达标线的对应系数列表；date是当前日期；title是图表的标题。

### 4. 图表展示

><img src="https://wx3.sinaimg.cn/mw690/00872OYVgy1ge03hyv6xsj35pk2xkngu.jpg" style="zoom:200%;" />

***

<p align='right'>Author : Junyu
<p align='right'>Date : 2020-04-20