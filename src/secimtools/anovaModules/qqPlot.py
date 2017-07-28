#Add-on packages
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib.backends.backend_pdf import PdfPages

# Plotting packages
from secimtools.visualManager import module_box as box
from secimtools.visualManager import module_hist as hist
from secimtools.visualManager import module_lines as lines
from secimtools.visualManager import module_scatter as scatter
from secimtools.visualManager.manager_color import colorHandler
from secimtools.visualManager.manager_figure import figureHandler

def qqPlot(tresid, tfit, oname):
    """ 
    Plot the residual diagnostic plots by sample.

    Output q-q plot, boxplots and distributions of the residuals. These plots
    will be used diagnose if residuals are approximately normal.

    :Arguments:
        :type tresid: pandas.Series
        :param tresid: Pearson normalized residuals. (transposed)
                        (residuals / sqrt(MSE))

        :type tfit: pandas DataFrame
        :param tfit: output of the ANOVA (transposed)

        :type oname: string
        :param oname: Name of the output file in pdf format.

    :Returns:
        :rtype: PDF
        :returns: Outputs a pdf file containing all plots.

    """
    #Open pdf
    with PdfPages(oname) as pdf:

        # Stablishing axisLayout
        axisLayout = [(0,0,1,1),(0,1,1,1),(0,2,1,1),(1,0,3,1)]

        # Start plotting
        for col in tresid.columns:
            #Creating figure
            fig = figureHandler(proj='2d',numAx=4,numRow=2,numCol=3,
                                arrangement=axisLayout)

            data = tresid[col].values.ravel()
            noColors = list()
            for j in range(0,len(data)):
                noColors.append('b')#blue
            df_data = pd.DataFrame(data)

            # Removing missing so that it will plot correctly.  
            mask_nan_data = np.isnan(data)
            data = data[~mask_nan_data]


            # Plot qqplot on axis 0
            sm.graphics.qqplot(data,fit=True,line='r',ax=fig.ax[0])


            # Plot boxplot on axis 1
            box.boxSeries(ser=data,ax=fig.ax[1])


            # Plot histogram on axis 2
            hist.quickHist(ax=fig.ax[2],dat=df_data,orientation='horizontal')


            # Plot scatterplot on axis 3
            scatter.scatter2D(ax=fig.ax[3],x=tfit[col], y=tresid[col],
                                colorList=list('b'))


            # Draw cutoff line for scatterplot on axis 3
            lines.drawCutoffHoriz(ax=fig.ax[3],y=0)


            # Format axis 0
            fig.formatAxis(figTitle=col,axnum=0,grid=False,showX=True,
                yTitle="Sample Quantiles", xTitle=" ")


            # Format axis 1
            fig.formatAxis(axnum=1,axTitle="Standardized Residuals",
                grid=False,showX=False,showY=True, xTitle=" ")

            # Format axis 2
            fig.formatAxis(axnum=2,grid=False,showX=True,showY=True,
                axTitle=" ",xTitle=" ")


            # Format axis 3
            fig.formatAxis(axnum=3,axTitle="Predicted Values vs Residual Values",
                xTitle="Predicted Values",yTitle="Residual Values",
                grid=False)


            #Add figure to pdf
            fig.addToPdf(pdfPages=pdf)
