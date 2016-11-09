######################################################################################
# DATE: 2016/August/10
# 
# MODULE: scatter3D.py
#
# VERSION: 1.0
# 
# AUTHOR: Matt Thoburn (mthoburn@ufl.edu) ed. Miguel Ibarra (miguelib@ufl.edu)
#
# DESCRIPTION: This is a standalone executable for graphing 3D scatter plots from wide data
#
################################################################################

# Import Build-In libraries
import argparse

# Import Add-On libraries
import pandas as pd
import module_scatter as scatter
from manager_color import colorHandler
from manager_figure import figureHandler
from argparse import RawDescriptionHelpFormatter

# Import Local libraries
from interface import wideToDesign

def getOptions(myOpts=None):
    description=""
    parser = argparse.ArgumentParser(description=description, 
                                    formatter_class=RawDescriptionHelpFormatter)
    standard = parser.add_argument_group(title='Standard input', 
                                description='Standard input for SECIM tools.')
    standard.add_argument( "-i","--input", dest="input", action='store', required=True, 
                        help="Input dataset in wide format.")
    standard.add_argument("-d" ,"--design",dest="design", action='store', required=False,
                        default=False,help="Design file.")
    standard.add_argument("-id", "--ID",dest="uniqID", action='store', required=True, 
                        help="Name of the column with unique identifiers.")
    standard.add_argument("-x" ,"--X",dest="x", action='store', required=True,
                        help="Name of column for X values")
    standard.add_argument("-y", "--Y",dest="y", action='store', required=True, 
                        help="Name of column for Y values")
    standard.add_argument("-z", "--Z",dest="z", action='store', required=True, 
                        help="Name of column for Z values")

    output = parser.add_argument_group(title='Required output')
    output.add_argument("-f","--figure",dest="figure",action="store",required=False,
                        help="Path of figure.")

    optional = parser.add_argument_group(title='Optional input')
    optional.add_argument("-g", "--group", dest="group", action='store', 
                        required=False, default=False, help="Name of the column "\
                        "with groups.")
    optional.add_argument("-pg", "--palette_group", dest="paletteGroup", action='store',
                        required=False, default="tableau", help="see palettable"\
                        " website.")
    optional.add_argument("-p", "--palette", dest="palette", action='store',
                        required=False, default="Tableau_20",help="see palettable"\
                        " website.")
    optional.add_argument("-r", "--rotation", dest="rotation", action='store',
                        required=False, default=45,help="camera viewing rotation")
    optional.add_argument("-e", "--elevation", dest="elevation", action='store',
                        required=False, default=45,help="Camera vieweing elevation")
    optional.add_argument("-pca", "--pca", dest="pca", action='store_true',
                        required=False, default=False,help="PCA ouput data or "\
                        "standard data")

    args = parser.parse_args()
    return(args)
     
def main(args):
    # Get design
    if args.design:
        design = pd.DataFrame.from_csv(args.design,sep="\t")
        design.reset_index(inplace=True)
    else:
        design = False

    # Assigning x,y,z values to easy acces
    X = args.x
    Y = args.y
    Z = args.z

    # Select wether you want to use output from PCA or just a Wide dataset
    if args.pca:
        wide = pd.DataFrame.from_csv(args.input,header=3,sep="\t")
    else:
        wide = pd.DataFrame.from_csv(args.input,sep="\t")

    # Open Figure handler
    fh = figureHandler(proj="3d")

    # Get colors
    if args.group and args.uniqID:
        glist=list(design[args.group])
        ch = colorHandler(args.paletteGroup,args.palette)
        colorList, ucGroups = ch.getColorsByGroup(design=design,group=args.group,
                                                    uGroup=sorted(set(glist)))
    else:
        glist = list()
        colorList = list('b') #blue
        ucGroups = dict()
    
    # Plot scatterplot
    scatter.scatter3D(ax=fh.ax[0],x=list(wide[X]),y=list(wide[Y]),z=list(wide[Z]),
                    colorList=colorList)

    # Give format to the plot
    fh.format3D(title=X + " vs " + Y + " vs " + Z, xTitle=X, yTitle=Y, zTitle=Z, 
                rotation=float(args.rotation),elevation=float(args.elevation))
    
    # Make legend
    if args.group and args.uniqID:
        fh.makeLegend(ax=fh.ax[0], ucGroups=ucGroups, group=args.group)
        fh.shrink()

    # Export figure
    fh.export(args.figure)

if __name__ == '__main__':
    # Command line options
    args = getOptions()

    main(args)