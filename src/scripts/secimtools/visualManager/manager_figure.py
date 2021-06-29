######################################################################################
# Date: 2016/June/03
#
# Module: manager_figure.py
#
# VERSION: 1.0
#
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu) ed. Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This module a class to manage figure and axes and their presentation
#
#######################################################################################
import warnings
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


class figureHandler:
    """class to manage figures, axes, and their presentation"""

    def export(self, out, dpi=90):
        """
        Exports figure to pdf with a given filename at a given resolution

        :Arguments:
            :type out: string
            :param out: path and filename of output pdf

            :type dpi: int
            :param dpi: resolution, measured in dots per square inch
        """
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        self.fig.set_rasterized(True)
        self.fig.savefig(out, dpi=dpi, format="pdf")

    def addToPdf(self, pdfPages, dpi=300):
        """
        Adds a figure to a pdfPages object, so that
        multiple figures can be printed in one document

        :Arguments:
            :type dpi: int
            :param dpi: resolution, measured in dots per square inch

            :type pdfPages: matplotlib pdfPages
            :param pdfPages: pdfPages object to be made into a document
        """
        # with warnings.catch_warnings():
        #    warnings.simplefilter("ignore")
        self.fig.set_rasterized(True)
        self.fig.savefig(pdfPages, dpi=dpi, format="pdf")
        plt.close(self.fig)

    def shrink(self, top=0.90, bottom=0.2, left=0.15, right=0.7):
        """
        Adjusts figure size to accomodate legends and axis set_ticks_position
        :type top: float
        :param top: adjustment value. Must be greater than bottom value

        :type bottom: float
        :param bottom: adjustment value. Must be less than top value

        :type left: float
        :param left: adjustment value. Must be less than right value

        :type right: float
        :param right: adjustment value. Must be greater than left value
        """

        self.fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)

    def despine(self, ax):
        """
        removes top and right axes from axis

        :Argument:
            :type ax: matplotlib Axis
            :param ax: axis to remove spines and ticks from
        """

        # Removing splines form top and right
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Remove xticks of top and right (but keeping bottom and left)
        ax.yaxis.set_ticks_position("none")
        ax.xaxis.set_ticks_position("none")

    # ylim and ylim are tuples of min,max
    def formatAxis(
        self,
        figTitle=None,
        axnum=0,
        xTitle="",
        yTitle="",
        axTitle="",
        xlim=None,
        ylim=None,
        grid=False,
        showX=True,
        showY=True,
        xticks=[],
    ):
        """
        Formats a given 2D axis

        :Arguemnts:
            :type figTitle: string
            :param figTitle: figure title at top of page

            :type axnum: int
            :param axnum: number of axis from figureHandler's axis list

            :type xTitle: string
            :param xTitle: x axis title

            :type yTitle: string
            :param yTitle: y axis title

            :type axTitle: string
            :param axTitle: axis title

            :type xlim: tuple of two ints
            :param xlim: minimum and maximum values for x axes

            :type ylim: tuple of two ints
            :param ylim: minimum and maximum values for y axes.

            :type xticks: list of strings
            :param xticks: names of xticks for graphs like bar graphs.

            :type grid: bool
            :param grid: if true, axis has grid, else axis has no grid

            :type showX: bool
            :param showX: if true, x axis are shown, otherwise x axis are hidden

            :type showY: bool
            :param showY: if true, y axis are shown, otherwise y axis are hidden
        """
        # Etablish figtitle
        if figTitle != None:
            self.fig.suptitle(figTitle, fontsize=12)

        # Set titles
        if xTitle != "":
            self.ax[axnum].set_xlabel(xTitle, fontweight="bold", fontsize=8)
        if yTitle != "":
            self.ax[axnum].set_ylabel(yTitle, fontweight="bold", fontsize=8)
        if axTitle != "":
            self.ax[axnum].set_title(axTitle, fontweight="bold", fontsize=8)
        self.ax[axnum].grid(grid)

        # Make top and right line invisible
        self.ax[axnum].get_xaxis().set_visible(showX)
        self.ax[axnum].get_yaxis().set_visible(showY)

        # Get limits for x and y, min and max
        xmin, xmax = self.ax[axnum].get_xlim()
        # print xmax
        ymin, ymax = self.ax[axnum].get_ylim()

        # Add personalized xticks
        if len(xticks) > 0:
            plt.xticks(list(range(len(xticks))), xticks, rotation="vertical")

        # Re-Orient the labels on x and y axis
        plt.setp(self.ax[axnum].xaxis.get_majorticklabels(), rotation=90)
        plt.setp(self.ax[axnum].yaxis.get_majorticklabels(), rotation=0)

        # Change xlimits
        if xlim == None:
            self.ax[axnum].set_xlim(xmin - abs(xmin) * 0.05, xmax + abs(xmax) * 0.05)
        elif xlim == "ignore":
            pass
        else:
            self.ax[axnum].set_xlim(xlim)

        # Change ylimits
        if ylim == None:
            self.ax[axnum].set_ylim(ymin - 0.025 * ymax, ymax + 0.025 * ymax)
        elif ylim == "ignore":
            pass
        else:
            self.ax[axnum].set_ylim(ylim)

        # Resize just a bit
        self.fig.subplots_adjust(top=0.9, bottom=0.2, left=0.15)

    def format3D(
        self, xTitle="", yTitle="", zTitle="", elevation=45, rotation=45, title=None
    ):
        """
        Formats a given 3D axis

        :Arguemnts:
            :type figTitle: string
            :param figTitle: figure title at top of page

            :type axnum: int
            :param axnum: number of axis from figureHandler's axis list

            :type xTitle: string
            :param xTitle: x axis title

            :type yTitle: string
            :param yTitle: y axis title

            :type zTitle: string
            :param zTitle: z axis title

            :type elevation: int
            :param elevation: camera elevation

            :type rotation: int
            :param rotation: camera rotation

            :type title: string
            :param title: axis title
        """

        if title == None:
            title = xTitle + " vs " + yTitle + " vs " + zTitle
        self.ax[0].set_xlabel(xTitle, fontweight="bold")
        self.ax[0].set_ylabel(yTitle, fontweight="bold")
        self.ax[0].set_zlabel(zTitle, fontweight="bold")
        self.ax[0].set_title(title, y=1.12, fontweight="bold")
        self.ax[0].grid(True)
        self.fig.subplots_adjust(top=0.93, left=0.05, right=0.7)
        self.ax[0].elev = float(elevation)
        self.ax[0].azim = float(rotation)

    def makeLegendLabel(self, ax):
        """
        Makes a legend using the label functionality of matplotlib

        :type ax: matplotlib axis
        :param ax: axis to draw legend on

        """

        # ax.legend(loc="upper left",frameon=True,bbox_to_anchor=(1.017,1.017))
        ax.legend(loc="upper left", frameon=True, bbox_to_anchor=(0.8, 1.0))

    def makeLegend(self, ax, ucGroups, group):
        """
        This function makes a legend based on the colors and/or markers and their groups

        :Arguments:
            :type ax: matplotlib Axis3D
            :param ax: Axis on which scatter plot will be drawn

            :type ucGroups: list of Strings
            :param ch: unique color groups

            :type group: string
            :param group: name of primary group

        :Return:
            :type ax: Matplotlib Axis
            :param ax: axis with legend plotted onto it
        """
        if len(ucGroups) >= 20:
            cols = 2
        else:
            cols = 1

        pltsColor = [
            matplotlib.lines.Line2D([0], [0], linestyle="none", c=color, marker="o")
            for name, color in sorted(ucGroups.items())
        ]
        colorLabels = [name for name, color in sorted(ucGroups.items())]

        # Print color legend
        # BBox to anchor works by a theoretical x,y coordinate relative to the figure
        # starting at a point specified by loc.
        colorLegend = ax.legend(
            pltsColor,
            colorLabels,
            loc="upper left",
            frameon=True,
            title=group,
            bbox_to_anchor=(1.1, 1.0),
            numpoints=1,
            ncol=cols,
            fontsize=8,
        )
        # Add legend to axis
        legendAdded = ax.add_artist(colorLegend)

        # Return axis
        return ax

    def __init__(
        self, proj, numAx=1, numRow=None, numCol=None, arrangement=None, figsize=None
    ):
        """
        Creates figureHandler with a list of axes and their arrangements

        :Arguments:
            :type proj: string
            :param proj: either 2d or 3d

            :type numAx: int
            :param numAx: number of axes in this figure

            :type numRow: int
            :param numRow: number of 'rows' in the figure's axes arrangment


            :type numCol: int
            :param numCol: number of 'columns' in the figure's axes arrangment

            :type arrangment: list of tuples of the form (x,y,colspan,rowspan)
            :param arrangement: matplotlib starts 'axis grids' with 0,0 at the top left corner and
                                go col,row. see pyplot.subplot2grid() for more details

            :type figsize:tuple
            :param figsize (optional): tuple with figure size

        :Returns:
            **Attributes**
            self.fig (matplotlib.figure): figure
            self.ax (list of matplotlib.axis): a list of one or more axes
        """
        if figsize:
            self.fig = plt.figure(figsize=figsize)
        else:
            self.fig = plt.figure()

        self.ax = list()

        if numAx == 1:
            if proj == "2d":
                self.ax.append(self.fig.add_subplot(111))
                self.ax[0].set_facecolor("w")
                self.despine(self.ax[0])
            else:
                self.ax.append(self.fig.add_subplot(111, projection=proj))
                self.ax[0].set_facecolor("w")
                self.despine(self.ax[0])
        else:
            for i in range(0, numAx):
                x = arrangement[i][0]
                y = arrangement[i][1]
                cs = arrangement[i][2]
                rs = arrangement[i][3]
                self.ax.append(
                    plt.subplot2grid((numRow, numCol), (x, y), rowspan=rs, colspan=cs)
                )
                for axis in self.ax:
                    axis.set_facecolor("w")
                    self.despine(axis)
