##载入所需包
import sys
sys.path.append('/home/junyu/PycharmProjects/work_plain/F1_example')
import achievement as achi
import pandas as pd

##载入数据
data = pd.read_csv('./PycharmProjects/work_plain/F1_example/data.csv')

##转为字典
data_dict = data.to_dict(orient='records')

##画图
c = achi.Achievement(data_dict)
c.Data_Index(x_name = '还款户数', y_name = '还款金额', size_name = ['阶段天数'], size_coef= [1.1], series = '地区', name = '处理人')
c.plot_axis(width = '1000px', height='600px', x_min=0, x_max=210, y_min=0, y_max=400000)
c.Plot(jine=[230000], hushu=[75, 120, 140, 160], coef=[0.9, 1.0, 1.1, 1.15], date = '2010-01-01', title = '绩效成绩').render_notebook()
