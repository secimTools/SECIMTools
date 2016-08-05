from manager_color import colorHandler
from manager_figure import figureHandler
from interface import wideToDesign
import module_scatter as scatter
import pandas

import argparse
from argparse import RawDescriptionHelpFormatter

def getOptions(myOpts=None):
    description=""
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', required=True, 
                        help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', required=False,default=False,
                        help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', required=True, 
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-x" ,"--X",dest="x", action='store', required=True,
                        help="Name of column for X values")
    standard.add_argument("-y", "--Y",dest="y", action='store', required=True, 
                        help="Name of column for Y values")

    

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-f","--figure",dest="figure",action="store",required=False,
                        help="Path of figure.")

    optional = parser.add_argument_group(title='Optional input')
    optional.add_argument("-g", "--group",dest="group", action='store', required=False, 
                        default=False,help="Name of the column with groups.")
    optional.add_argument("-pg","--palette_group",dest="paletteGroup",action='store',
                        required=False,default="tableau",help="See palettable website")
    optional.add_argument("-p","--palette",dest="palette",action='store',
                        required=False,default="Tableau_20",help="See palettable website")
    standard.add_argument("-pca","--pca",dest="pca",action='store_true',
                    required=False,default=False,help="PCA ouput data or standard data")

    args = parser.parse_args()
    return(args)

def main(args):
    data = args.input
    if args.design:
        design = args.design
        design = pandas.DataFrame.from_csv(design,sep="\t")
        design.reset_index(inplace=True)
    else:
        design = False
    uniqID = args.uniqID
    group = args.group

    out = args.figure

    X = args.x
    Y = args.y

    paletteGroup = args.paletteGroup
    palette = args.palette

    usePCA = args.pca
    print usePCA

    if usePCA:
        wide = pandas.DataFrame.from_csv(data,header=3,sep="\t")
    else:
        wide = pandas.DataFrame.from_csv(data,sep="\t")

    
    fh = figureHandler(proj="2d")


    if group and uniqID != "rowID":
        glist=list(design[group])
        ch = colorHandler(paletteGroup,palette)
        colorList, ucGroups = ch.getColorsByGroup(design=design,group=group,uGroup=sorted(set(glist)))
    else:
        glist = list()
        colorList = list('b') #blue
        ucGroups = dict()
        
    scatter.scatter2D(ax=fh.ax[0],x=list(wide[X]),y=list(wide[Y]),colorList=colorList)

    fh.despine(fh.ax[0])
    xlim = fh.ax[0].get_xlim()
    ylim = fh.ax[0].get_ylim()
    fh.formatAxis(figTitle=X + " vs " + Y, xTitle=X, yTitle=Y, grid=False,xlim=xlim,ylim=ylim)
    
    if group and uniqID != "rowID":
        fh.makeLegend(ax=fh.ax[0],ucGroups=ucGroups,group=group)
        fh.shrink()

    fh.export(out)

if __name__ == '__main__':
    # Command line options
    args = getOptions()


    main(args)