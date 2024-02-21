import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import math as math

class Graphs:

    def __init__(self) -> None:
        pass

    def graphCorrelation(self, df : pd.DataFrame, title = "Correlation HeatMap"):
        plt.figure("Correalation HeatMap")
        corr = df.corr(numeric_only=True).round(2)
        sb.heatmap(corr, annot=True)
        plt.title(title, fontsize = 15)
        plt.show()
    
    def basicPlotForOneField(self, y : list, title = "Basic Plot", x_label = "None", y_label = "None"):
        plt.figure("Basic Plot For One Column")
        plt.plot(y)
        plt.title(title, fontsize=15)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show()

    def basicPlotForTwoFields(self, x : list, y : list, title = "Basic Plot", x_label = "None", y_label = "None"):
        plt.figure("Basic Plot for Two Columns")
        plt.plot(x, y)
        plt.title(title, fontsize=15)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show() 

    def distPlotWithSubPlot(self, df : pd.DataFrame, features : list = [], rows : int = 0, col : int = 0):
        plt.figure("DistPlot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                sb.distplot(df[fch])
                plt.xlabel(fch)
        else:
            if(rows * col < len(features)):
                raise Exception("Invalid SubPlot Size")
            for index, fch in enumerate(features):
                plt.subplot(rows, col, index+1)
                sb.distplot(df[fch])
                plt.xlabel(fch)
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()

    def barplotWithSubplot(self, df : pd.DataFrame, features : list = [], rows : int = 0, col : int = 0):
        plt.figure("BoxPlot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                df[fch].plot.bar()
                plt.xlabel(fch)
        else:
            if(rows * col < len(features)):
                raise Exception("Invalid SubPlot Size")
            for index, fch in enumerate(features):
                plt.subplot(rows, col, index+1)
                df[fch].plot.bar()
                plt.xlabel(fch)
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()

    def scatterPlotWithSubPlot(self, df: pd.DataFrame, target: str, features: list = [],  rows: int = 0, col: int = 0):
        plt.figure("Scatter Plot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            features.remove(target)
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                plt.scatter(df[fch], df[target])
                plt.xlabel(fch)
                plt.ylabel(target)
                plt.title(f'Scatter plot between {fch} and {target} ')
        else:
            if(rows * col < len(features)):
                raise Exception("Invalid SubPlot Size")
            for index, fch in enumerate(features):
                plt.subplot(rows, col, index+1)
                plt.scatter(df[fch], df[target])
                plt.xlabel(fch)
                plt.ylabel(target)
                plt.title(f'Scatter plot between {fch} and {target} ')
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()


    def scatterPlotWithSubPlotAndBestFitLine(self, df: pd.DataFrame, target: str, features: list = [],  rows: int = 0, col: int = 0):
        plt.figure("Scatter Plot's with subPlot and bestfitline")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            features.remove(target)
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                plt.scatter(df[fch], df[target])
                m, c = np.polyfit(df[fch], df[target],deg= 1) 
                plt.plot(df[fch], m*df[fch]+c, color = 'red')
                plt.xlabel(fch)
                plt.ylabel(target)
                plt.title(f'Scatter plot between {fch} and {target}  and bestfitline')
        else:
            if(rows * col < len(features)):
                raise Exception("Invalid SubPlot Size")
            for index, fch in enumerate(features):
                plt.subplot(rows, col, index+1)
                plt.scatter(df[fch], df[target])
                m, c = np.polyfit(df[fch], df[target],deg= 1) 
                plt.plot(df[fch], m*df[fch]+c, color = 'red')
                plt.xlabel(fch)
                plt.ylabel(target)
                plt.title(f'Scatter plot between {fch} and {target} and bestfitline')
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()


    def boxPlotWithSubplot(self, df: pd.DataFrame, features: list = [],  rows: int = 0, col: int = 0, wisker_length : float = 1.5):
        plt.figure("Boxplot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                plt.boxplot(df[fch], vert=False, whis=wisker_length)
                plt.title(f'boxplot of {fch}')
        else:
            if(rows * col < len(features)):
                raise Exception("Invalid SubPlot Size")
            for index, fch in enumerate(features):
                plt.subplot(rows, col, index+1)
                plt.boxplot(df[fch], vert=False, whis=wisker_length)
                plt.title(f'boxplot of {fch}')
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()