import pandas as pd
import numpy as np
import imageio
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from pyecharts import options as opts
from pyecharts.charts import Map
from pyecharts.render import make_snapshot
from snapshot_phantomjs import snapshot
from pyecharts.charts import Geo
from pyecharts.globals import ChartType
from mpl_toolkits.mplot3d import Axes3D


def job1():
    plt.rcParams['figure.figsize'] = (12.0, 12.0)
    # 读取数据
    iris = pd.read_csv('iris.csv')
    print(iris)
    colors = ['r', 'y', 'b']  # 定义三种散点的颜色
    Species = iris.Species.unique()  # 对类别去重
    print(Species)
    order1=['Sepal.length','Sepal.width','Petal.Length','Petal.Width']
    order2=['Petal.Width','Petal.Length','Sepal.width','Sepal.length']
    for r in range(4):
        for c in range(4):
            plt.subplot(4,4,4*r+c+1)
            for i in range(len(Species)):
                plt.scatter(iris.loc[iris.Species == Species[i], order1[r]],
                            iris.loc[iris.Species == Species[i], order2[c]],s=20 ,c=colors[i], label=Species[i])
                # 添加轴标签和标题
            plt.title(order1[r]+' vs '+order2[c])
            plt.xlabel(order1[r])
            plt.ylabel(order2[c])
            plt.grid(True, linestyle='--', alpha=0.8)  # 设置网格线
            plt.style.use('seaborn-bright')
    plt.legend(bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0) #设置图例在图外右下角
    plt.show()

def job2():
    df=pd.read_csv('train.csv',usecols=['Age'])
    df.dropna()#去除空数据
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']#设置中文字体

    #sections = [0, 10, 20, 30, 40, 50, 60, 70, 80]  # 设置年龄分段
    #sections_name = ['0-10岁', '10-20岁', '20-30岁','30-40岁','40-50岁','50-60岁','60-70岁', '70-80岁']  # 年龄分段标签

    sections=[0,20,40,60,80]#设置年龄分段
    sections_name = ['0-20岁', '20-40岁', '40-60岁', '60-80岁']#年龄分段标签
    result = pd.cut(df.Age, sections, labels=sections_name)
    count = pd.value_counts(result, sort=False)#计算每个年龄人数list
    plt.title("泰坦尼克号乘客各年龄段人数")
    plt.pie(count, labels=sections_name, labeldistance=1.2, autopct="%1.2f%%", shadow=True,
            startangle=0, pctdistance=0.5)
    plt.show()

def job3():
    df=pd.read_csv('中国大学数量.csv')

    class Data:
        provinces = df.iloc[1:, 0].values.tolist()#省份
        @staticmethod
        def values() -> list:
            return df.iloc[1:,4].values.tolist()#公办本科大学数量列

    def map_visualmap() -> Map:
        c = (
            Map()
            .add("公办本科大学数量", [list(z) for z in zip(Data.provinces, Data.values())], "china")
            .set_global_opts(
                title_opts=opts.TitleOpts(title="公办本科大学数量"),
                visualmap_opts=opts.VisualMapOpts(min_=0,max_=70))
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
        )
        return c
    map_visualmap().render("map.html")

def job4():
    class Data:
        city = ["淮南市"]
        values=[[3],[2],[2],[5],[5],[5],[1],[3],[-1],[-4]]

    def geo_hometown(title,i) -> Geo:
        c = (Geo()
            .add_schema(maptype="淮南")
            .add(
            title, [list(z) for z in zip(Data.city, Data.values[i])],
            type_=ChartType.HEATMAP)
            .set_global_opts(
            visualmap_opts=opts.VisualMapOpts(min_=-4 ,max_=5, is_piecewise=True),
            title_opts=opts.TitleOpts(title="淮南市近十天温度变化情况"),
        )
        )
        return c

    for i in range(10):
        str_date = "12月" + str(i + 5) + "日"
        geo_hometown(str_date,i).render(str_date + ".html")
        make_snapshot(snapshot, geo_hometown(str_date,i).render(),
                      str(i + 1) + ".png", pixel_ratio=1)

    frames=[]
    for i in range(10):
        frames.append(imageio.imread(str(i+1)+".png"))
    imageio.mimsave("huainan.gif", frames, 'GIF', duration=0.35)

def job5():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    x = np.array([0, 1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1, 0, 0, 0, 0, 1, 2, 3, 3, 3, 2, 1, 1, 2])  # 生成x轴的数据
    y = np.array([0, 0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4, 4, 3, 2, 1, 1, 1, 1, 2, 3, 3, 3, 2, 2])  # 生成y轴的数据
    z = np.array([i for i in range(1, 26)])  # 生成z轴的数据

    print('=========================================bottom')
    bottom = np.zeros_like(z)  # 产生一个全零的矩阵，形状和z一样。bottom表示直方图从哪个数值开始是底部

    width = depth = 0.8  # width和depth表示直方柱的宽度和深度在单元格中比例
    # 3.调用bar3d，画3D直方图
    ax.bar3d(x, y, bottom, width, depth, z, shade=True)

    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 1, 2, 3, 4, 5])

    # 4.显示图形
    plt.show()

def job6():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=10,azim=25)
    plt.yticks(fontsize=8)
    plt.xticks(fontsize=8)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 2.生成数据
    x1 = np.array([-1,0,1,-1,-1,0,1,-1,1,1])  # 生成x轴的数据
    y1 = np.array([-1,0,-1,-1,1,0,1,1,1,-1])  # 生成y轴的数据
    z1 = np.array([0,0.8,0,0,0,0.8,0,0,0,0])
    z2 = np.array([0,-0.8,0,0,0,-0.8,0,0,0,0])
    # 3.调用plot，画3D的线图
    ax.plot(x1, y1, z2, "b")
    ax.plot(x1, y1, z1, "r")

    plt.show()

def job7():
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=10, azim=30)
    ax.set_title('scatter diagram')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # 2.生成数据
    _x1=list(range(-100,101,2))*100
    x1 = np.array(_x1)  # 生成x轴的数据
    _y1 = []
    for i in range(-100, 101,2):
        for j in range(100):
            _y1.append(i)
    y1 = np.array(_y1)  # 生成y轴的数据
    z1 = 20000-x1*x1-y1*y1
    z2 = x1*x1+y1*y1-20000
    # 3.调用plot，画3D的线图
    ax.scatter(x1, y1, z1,s=2, c="b")
    ax.scatter(x1,y1,z2,s=2,c="r")
    plt.show()


if __name__ == '__main__':
    job1()
    job2()
    job3()
    job4()
    job5()
    job6()
    job7()

