from pyecharts import options as opts
from pyecharts.charts import Scatter
from pyecharts.globals import ThemeType


class Achievement():
    
    def __init__(self, data: [dict]):
        self.num = len(data)
        data_value = [tuple(item.values()) for item in data]
        self.data_value = data_value
        data_name = tuple(data[0].keys())
        self.data_name = data_name
        
    def Data_Index(self, x_name: str, y_name: str, size_name: [str], size_coef: [float], series: str, name: str):
        self.x_index = self.data_name.index(x_name)
        self.y_index = self.data_name.index(y_name)
        self.size_index = [self.data_name.index(i) for i in size_name]
        self.size_coef = size_coef
        self.local_index = self.data_name.index(series)
        self.name_index = self.data_name.index(name)
        return self
        
    def plot_axis(self, width, height, x_min, x_max, y_min, y_max, theme = ThemeType.LIGHT):
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.theme = theme
        return self
    
    def Plot(self, jine: list, hushu: list, coef: list, date :str = None, title : str = None, size_def = lambda x: 50 if 10 + x > 50 else 10 + x):
        
        ##配置全局变量
        j_list = [0 for _ in jine]
        h_list = [0 for _ in hushu]
        plot = Scatter(init_opts = opts.InitOpts(width = self.width, height = self.height, theme = self.theme))
        plot.set_global_opts(
                            title_opts=opts.TitleOpts(title=title),
                            xaxis_opts = opts.AxisOpts(type_="value", 
                                                       splitline_opts=opts.SplitLineOpts(is_show=False), 
                                                       min_ = self.x_min, max_ = self.x_max, 
                                                       name = self.data_name[self.x_index]),
                            yaxis_opts = opts.AxisOpts(type_="value", 
                                                       splitline_opts=opts.SplitLineOpts(is_show=False), 
                                                       min_ = self.y_min, max_ = self.y_max, 
                                                       name = self.data_name[self.y_index]),
                            toolbox_opts = opts.ToolboxOpts(orient = 'horizontal', pos_top = 0,
                                                            feature = opts.ToolBoxFeatureOpts(
                                                            save_as_image = opts.ToolBoxFeatureSaveAsImageOpts(type_ = "jpeg", pixel_ratio = 4, background_color = 'white'),
                                                            restore = opts.ToolBoxFeatureRestoreOpts(),
                                                            data_zoom = opts.ToolBoxFeatureDataZoomOpts(),
                                                            data_view = opts.ToolBoxFeatureDataViewOpts(is_show = False),
                                                            magic_type = opts.ToolBoxFeatureMagicTypeOpts(is_show = False),
                                                            brush = opts.ToolBoxFeatureBrushOpts(type_=[]))),
                            tooltip_opts=opts.TooltipOpts(is_show=True, trigger="item", axis_pointer_type="cross")
                            )
        
        ##逐个添加点
        for item in self.data_value:
            
            #数据处理
            local = str(item[self.local_index])
            num = float(item[self.x_index])
            gold = float(item[self.y_index])
            name = str(item[self.name_index])
            size = 0.
            for s in range(len(self.size_index)):
                size += item[self.size_index[s]]*self.size_coef[s]
            tips = '日期:{}'.format(date)
            for s in range(len(self.data_name)):
                tips += '<br>'
                tips += '{}:{}'.format(self.data_name[s], item[s])
            
            #添加点及其tips
            plot.add_xaxis([num])
            plot.add_yaxis(local, [gold], symbol_size = size_def(size),
                           tooltip_opts=opts.TooltipOpts(formatter = tips),
                           label_opts=opts.LabelOpts(formatter = name))
            
            #计算达标人数
            for i in range(len(j_list)):
                if gold >= jine[i]:
                    j_list[i] += 1
            for i in range(len(h_list)):
                if num >= hushu[i]:
                    h_list[i] += 1
                    
        ##添加达标线及其注释点
        for item in range(len(j_list)):
            plot.add_xaxis([self.x_max])
            plot.add_yaxis('达标线', [jine[item]], symbol_size = 25, symbol = 'pin',
                           label_opts = opts.LabelOpts(is_show = False),
                           tooltip_opts = opts.TooltipOpts(formatter = '达标人数:{}<br>未达标人数:{}'.format(j_list[item], self.num-j_list[item])),
                           markline_opts = opts.MarkLineOpts(data = [opts.MarkLineItem(y = jine[item], name = '金额达标线{}'.format(i+1))],
                                                             label_opts = opts.LabelOpts(is_show = False),
                                                             symbol_size = 0))
            
        for item in range(len(h_list)):
            plot.add_xaxis([hushu[item]])
            plot.add_yaxis('达标线', [self.y_max], symbol_size = 25, symbol = 'pin',
                           label_opts = opts.LabelOpts(is_show = False),
                           tooltip_opts = opts.TooltipOpts(formatter = '奖励系数:x{}<br>达标人数:{}<br>未达标人数:{}'.format(coef[item], h_list[item], self.num-h_list[item])),
                           markline_opts = opts.MarkLineOpts(data = [opts.MarkLineItem(x = hushu[item], name = '户数达标线{}'.format(i+1))],
                                                             label_opts = opts.LabelOpts(is_show = False),
                                                             symbol_size = 0)) 
        
        return plot
