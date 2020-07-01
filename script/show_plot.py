# coding=utf-8
import matplotlib
##matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

## 一图中2条线
def show_two_plot():
    x = np.linspace(0, 2*np.pi, 50)
    plt.plot(x, np.sin(x), 'r-o',
            x, np.sin(2*x), 'g--' )
    plt.show()

## 子图
def show_plot():
    x = np.linspace(0, 2*np.pi, 50)
    # (界面划分子图的行数， 界面划分子图的列数， 要绘图的那个活跃区的序号从1开始)
    plt.subplot(2, 1, 1)
    plt.plot(x, np.sin(x), 'r')

    plt.subplot(2, 1, 2)
    plt.plot(x, np.cos(x), 'g')
    plt.show()

## 散点图
def show_scatter():
    x = np.linspace(0, 2*np.pi, 50)
    y = np.sin(x)
    plt.scatter(x, y)
    plt.show()

## 散点图,设置大小和颜色
def show_scatter_with_size_color():
    x = np.random.rand(1000)
    y = np.random.rand(1000)
    size = np.random.rand(1000) * 50
    color = np.random.rand(1000)
    plt.scatter(x, y, size, color)
    plt.colorbar()
    plt.show()

## 直方图
def show_hist():
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    plt.hist([x, y], bins=50)
    plt.show()

## 添加标题、标签和图例
def show_title():
    x = np.linspace(0, 2*np.pi, 50)
    plt.plot(x, np.sin(x), 'r-x', label='Sin(x)')
    plt.plot(x, np.cos(x), 'g-^', label='Cos(x)')
    plt.legend() # 展示图例
    plt.xlabel('Rads') # 给x轴添加标签
    plt.ylabel('Amplitude') #给y轴添加标签
    plt.title('Sin and Cos Waves') #添加图形标题
    plt.show()


if __name__ == '__main__':
    ##show_two_plot()
    ##show_plot()
    ##show_scatter()
    ##show_scatter_with_size_color()
    ##show_hist()
    show_title()

