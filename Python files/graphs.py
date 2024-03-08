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
        corr = df.corr(numeric_only=True, method= "spearman").round(2)
        sb.heatmap(corr, annot=True)
        plt.title(title, fontsize = 15)
        plt.show()
    
    def basicPlot(self, y : list, x: list = None, title = "Basic Plot", x_label = "None", y_label = "None"):
        plt.figure("Basic Plot for One or Two Columns", figsize=(10, 5))
        if x is None:
            plt.plot(y)
        else:
            plt.plot(x, y)
        plt.title(title, fontsize=15)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.show() 

    def distPlotWithSubPlot(self, df : pd.DataFrame, features : list = [], rows : int = 0, cols : int = 0):
        plt.figure("DistPlot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                sb.distplot(df[fch])
                plt.xlabel(fch)
        else:
            if(rows * cols < len(features)):
                rows , cols = self.getRowAndColumnSize(features)
            for index, fch in enumerate(features):
                plt.subplot(rows, cols, index+1)
                sb.distplot(df[fch])
                plt.xlabel(fch)
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()

    def barplotWithSubplot(self, df : pd.DataFrame, features : list = [], rows : int = 0, cols : int = 0):
        plt.figure("BarPlot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            # Iterate over each column and its corresponding index
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                df[fch].plot.bar()
                plt.xlabel(fch)
        else:
            if(rows * cols < len(features)):
                rows , cols = self.getRowAndColumnSize(features)
            # Iterate over each column and its corresponding index
            for index, fch in enumerate(features):
                # Create subplots in a rowsxcols grid, with each subplot representing one of the numeric columns
                plt.subplot(rows, cols, index+1)
                df[fch].plot.bar()
                plt.xlabel(fch)
        # plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()

    def scatterPlotWithSubPlot(self, df: pd.DataFrame, target: str, features: list = [],  rows: int = 0, col: int = 0):
        plt.figure("Scatter Plot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            if(target in features):
                features.remove(target)
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                plt.scatter(df[fch], df[target])
                plt.xlabel(fch)
                plt.ylabel(target)
                plt.title(f'Scatter plot between {fch} and {target} ')
        else:
            if(rows * col < len(features)):
                rows , col = self.getRowAndColumnSize(features)
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
            if(target in features):
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
                rows , col = self.getRowAndColumnSize(features)
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


    def boxPlotWithSubplot(self, df: pd.DataFrame, features: list = [],  rows: int = 0, cols: int = 0, wisker_length : float = 1.5):
        plt.figure("Boxplot's with subPlot")
        if(len(features) == 0):
            features = df.select_dtypes(include=np.number).columns.tolist()
            for index, fch in enumerate(features):
                plt.subplot(math.ceil(len(features)/2), 2, index+1)
                plt.boxplot(df[fch], vert=False, whis=wisker_length)
                plt.title(f'boxplot of {fch}')
        else:
            if(rows * cols < len(features)):
                rows , cols = self.getRowAndColumnSize(features)
            for index, fch in enumerate(features):
                plt.subplot(rows, cols, index+1)
                plt.boxplot(df[fch], vert=False, whis=wisker_length)
                plt.title(f'boxplot of {fch}')
        plt.subplots_adjust(left=0.1, bottom=0.08, right=0.9, top=0.9, wspace=0.1, hspace=0.4)
        plt.show()


    def getRowAndColumnSize(self, features : list):
        rows = math.ceil(len(features)/2)
        col = 2
        return (rows, col)
    
    def dynamincSubPlotLogic(length : int):     
        if(length == 1):
            return (1, 1)
        elif(length == 2):
            return (2, 1)
        else:
            default_diviser , min_row, min_col = 2, 0, 0
            while(default_diviser <= int(math.ceil(length/2))):
                q = int(math.ceil(length / default_diviser))
                if((min_row + min_col > default_diviser + q) or (min_row == 0 and min_col == 0)):
                    min_row = default_diviser
                    min_col = q
                default_diviser = default_diviser + 1
            return ( min_row if min_row > min_col else min_col, min_row if min_row < min_col else min_col)    