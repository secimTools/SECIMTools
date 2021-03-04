######################################################################################
# Date: 2016/June/03
# 
# Module: manager_color.py
#
# VERSION: 1.0
# 
# AUTHOR: Miguel Ibarra (miguelib@ufl.edu) ed. Matt Thoburn (mthoburn@ufl.edu) 
#
# DESCRIPTION: This module a class to manage color for use in plotting
#
#######################################################################################
import palettable
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import palettable.tableau as tb
import palettable.wesanderson as wes
import palettable.cubehelix as cubhx
import palettable.colorbrewer.sequential as seq
import palettable.colorbrewer.diverging as div
import palettable.colorbrewer.qualitative as qual

class colorHandler:
    """class to handle colors used in graphing""" 
    def __init__(self,pal,col): 
        #All color palettes in palettable
        palettes={
            "diverging": {
                "BrBG_10": div.BrBG_10,
                "BrBG_11": div.BrBG_11,
                "BrBG_3": div.BrBG_3,
                "BrBG_4": div.BrBG_4,
                "BrBG_5": div.BrBG_5,
                "BrBG_6": div.BrBG_6,
                "BrBG_7": div.BrBG_7,
                "BrBG_8": div.BrBG_8,
                "BrBG_9": div.BrBG_9,
                "PRGn_10": div.PRGn_10,
                "PRGn_11": div.PRGn_11,
                "PRGn_3": div.PRGn_3,
                "PRGn_4": div.PRGn_4,
                "PRGn_5": div.PRGn_5,
                "PRGn_6": div.PRGn_6,
                "PRGn_7": div.PRGn_7,
                "PRGn_8": div.PRGn_8,
                "PRGn_9": div.PRGn_9,
                "PiYG_10": div.PiYG_10,
                "PiYG_11": div.PiYG_11,
                "PiYG_3": div.PiYG_3,
                "PiYG_4": div.PiYG_4,
                "PiYG_5": div.PiYG_5,
                "PiYG_6": div.PiYG_6,
                "PiYG_7": div.PiYG_7,
                "PiYG_8": div.PiYG_8,
                "PiYG_9": div.PiYG_9,
                "PuOr_10": div.PuOr_10,
                "PuOr_11": div.PuOr_11,
                "PuOr_3": div.PuOr_3,
                "PuOr_4": div.PuOr_4,
                "PuOr_5": div.PuOr_5,
                "PuOr_6": div.PuOr_6,
                "PuOr_7": div.PuOr_7,
                "PuOr_8": div.PuOr_8,
                "PuOr_9": div.PuOr_9,
                "RdBu_10": div.RdBu_10,
                "RdBu_11": div.RdBu_11,
                "RdBu_3": div.RdBu_3,
                "RdBu_4": div.RdBu_4,
                "RdBu_5": div.RdBu_5,
                "RdBu_6": div.RdBu_6,
                "RdBu_7": div.RdBu_7,
                "RdBu_8": div.RdBu_8,
                "RdBu_9": div.RdBu_9,
                "RdGy_10": div.RdGy_10,
                "RdGy_11": div.RdGy_11,
                "RdGy_3": div.RdGy_3,
                "RdGy_4": div.RdGy_4,
                "RdGy_5": div.RdGy_5,
                "RdGy_6": div.RdGy_6,
                "RdGy_7": div.RdGy_7,
                "RdGy_8": div.RdGy_8,
                "RdGy_9": div.RdGy_9,
                "RdYlBu_10": div.RdYlBu_10,
                "RdYlBu_11": div.RdYlBu_11,
                "RdYlBu_3": div.RdYlBu_3,
                "RdYlBu_4": div.RdYlBu_4,
                "RdYlBu_5": div.RdYlBu_5,
                "RdYlBu_6": div.RdYlBu_6,
                "RdYlBu_7": div.RdYlBu_7,
                "RdYlBu_8": div.RdYlBu_8,
                "RdYlBu_9": div.RdYlBu_9,
                "RdYlGn_10": div.RdYlGn_10,
                "RdYlGn_11": div.RdYlGn_11,
                "RdYlGn_3": div.RdYlGn_3,
                "RdYlGn_4": div.RdYlGn_4,
                "RdYlGn_5": div.RdYlGn_5,
                "RdYlGn_6": div.RdYlGn_6,
                "RdYlGn_7": div.RdYlGn_7,
                "RdYlGn_8": div.RdYlGn_8,
                "RdYlGn_9": div.RdYlGn_9,
                "Spectral_10": div.Spectral_10,
                "Spectral_11": div.Spectral_11,
                "Spectral_3": div.Spectral_3,
                "Spectral_4": div.Spectral_4,
                "Spectral_5": div.Spectral_5,
                "Spectral_6": div.Spectral_6,
                "Spectral_7": div.Spectral_7,
                "Spectral_8": div.Spectral_8,
                "Spectral_9": div.Spectral_9
            },
            "qualitative": {
                "Accent_3": qual.Accent_3,
                "Accent_4": qual.Accent_4,
                "Accent_5": qual.Accent_5,
                "Accent_6": qual.Accent_6,
                "Accent_7": qual.Accent_7,
                "Accent_8": qual.Accent_8,
                "Dark2_3": qual.Dark2_3,
                "Dark2_4": qual.Dark2_4,
                "Dark2_5": qual.Dark2_5,
                "Dark2_6": qual.Dark2_6,
                "Dark2_7": qual.Dark2_7,
                "Dark2_8": qual.Dark2_8,
                "Paired_10": qual.Paired_10,
                "Paired_11": qual.Paired_11,
                "Paired_12": qual.Paired_12,
                "Paired_3": qual.Paired_3,
                "Paired_4": qual.Paired_4,
                "Paired_5": qual.Paired_5,
                "Paired_6": qual.Paired_6,
                "Paired_7": qual.Paired_7,
                "Paired_8": qual.Paired_8,
                "Paired_9": qual.Paired_9,
                "Pastel1_3": qual.Pastel1_3,
                "Pastel1_4": qual.Pastel1_4,
                "Pastel1_5": qual.Pastel1_5,
                "Pastel1_6": qual.Pastel1_6,
                "Pastel1_7": qual.Pastel1_7,
                "Pastel1_8": qual.Pastel1_8,
                "Pastel1_9": qual.Pastel1_9,
                "Pastel2_3": qual.Pastel2_3,
                "Pastel2_4": qual.Pastel2_4,
                "Pastel2_5": qual.Pastel2_5,
                "Pastel2_6": qual.Pastel2_6,
                "Pastel2_7": qual.Pastel2_7,
                "Pastel2_8": qual.Pastel2_8,
                "Set1_3": qual.Set1_3,
                "Set1_4": qual.Set1_4,
                "Set1_5": qual.Set1_5,
                "Set1_6": qual.Set1_6,
                "Set1_7": qual.Set1_7,
                "Set1_8": qual.Set1_8,
                "Set1_9": qual.Set1_9,
                "Set2_3": qual.Set2_3,
                "Set2_4": qual.Set2_4,
                "Set2_5": qual.Set2_5,
                "Set2_6": qual.Set2_6,
                "Set2_7": qual.Set2_7,
                "Set2_8": qual.Set2_8,
                "Set3_10": qual.Set3_10,
                "Set3_11": qual.Set3_11,
                "Set3_12": qual.Set3_12,
                "Set3_3": qual.Set3_3,
                "Set3_4": qual.Set3_4,
                "Set3_5": qual.Set3_5,
                "Set3_6": qual.Set3_6,
                "Set3_7": qual.Set3_7,
                "Set3_8": qual.Set3_8,
                "Set3_9": qual.Set3_9
            },
            "sequential": { 
                "Blues_3": seq.Blues_3,
                "Blues_4": seq.Blues_4,
                "Blues_5": seq.Blues_5,
                "Blues_6": seq.Blues_6,
                "Blues_7": seq.Blues_7,
                "Blues_8": seq.Blues_8,
                "Blues_9": seq.Blues_9,
                "BuGn_3": seq.BuGn_3,
                "BuGn_4": seq.BuGn_4,
                "BuGn_5": seq.BuGn_5,
                "BuGn_6": seq.BuGn_6,
                "BuGn_7": seq.BuGn_7,
                "BuGn_8": seq.BuGn_8,
                "BuGn_9": seq.BuGn_9,
                "BuPu_3": seq.BuPu_3,
                "BuPu_4": seq.BuPu_4,
                "BuPu_5": seq.BuPu_5,
                "BuPu_6": seq.BuPu_6,
                "BuPu_7": seq.BuPu_7,
                "BuPu_8": seq.BuPu_8,
                "BuPu_9": seq.BuPu_9,
                "GnBu_3": seq.GnBu_3,
                "GnBu_4": seq.GnBu_4,
                "GnBu_5": seq.GnBu_5,
                "GnBu_6": seq.GnBu_6,
                "GnBu_7": seq.GnBu_7,
                "GnBu_8": seq.GnBu_8,
                "GnBu_9": seq.GnBu_9,
                "Greens_3": seq.Greens_3,
                "Greens_4": seq.Greens_4,
                "Greens_5": seq.Greens_5,
                "Greens_6": seq.Greens_6,
                "Greens_7": seq.Greens_7,
                "Greens_8": seq.Greens_8,
                "Greens_9": seq.Greens_9,
                "Greys_3": seq.Greys_3,
                "Greys_4": seq.Greys_4,
                "Greys_5": seq.Greys_5,
                "Greys_6": seq.Greys_6,
                "Greys_7": seq.Greys_7,
                "Greys_8": seq.Greys_8,
                "Greys_9": seq.Greys_9,
                "OrRd_3": seq.OrRd_3,
                "OrRd_4": seq.OrRd_4,
                "OrRd_5": seq.OrRd_5,
                "OrRd_6": seq.OrRd_6,
                "OrRd_7": seq.OrRd_7,
                "OrRd_8": seq.OrRd_8,
                "OrRd_9": seq.OrRd_9,
                "Oranges_3": seq.Oranges_3,
                "Oranges_4": seq.Oranges_4,
                "Oranges_5": seq.Oranges_5,
                "Oranges_6": seq.Oranges_6,
                "Oranges_7": seq.Oranges_7,
                "Oranges_8": seq.Oranges_8,
                "Oranges_9": seq.Oranges_9,
                "PuBuGn_3": seq.PuBuGn_3,
                "PuBuGn_4": seq.PuBuGn_4,
                "PuBuGn_5": seq.PuBuGn_5,
                "PuBuGn_6": seq.PuBuGn_6,
                "PuBuGn_7": seq.PuBuGn_7,
                "PuBuGn_8": seq.PuBuGn_8,
                "PuBuGn_9": seq.PuBuGn_9,
                "PuBu_3": seq.PuBu_3,
                "PuBu_4": seq.PuBu_4,
                "PuBu_5": seq.PuBu_5,
                "PuBu_6": seq.PuBu_6,
                "PuBu_7": seq.PuBu_7,
                "PuBu_8": seq.PuBu_8,
                "PuBu_9": seq.PuBu_9,
                "PuRd_3": seq.PuRd_3,
                "PuRd_4": seq.PuRd_4,
                "PuRd_5": seq.PuRd_5,
                "PuRd_6": seq.PuRd_6,
                "PuRd_7": seq.PuRd_7,
                "PuRd_8": seq.PuRd_8,
                "PuRd_9": seq.PuRd_9,
                "Purples_3": seq.Purples_3,
                "Purples_4": seq.Purples_4,
                "Purples_5": seq.Purples_5,
                "Purples_6": seq.Purples_6,
                "Purples_7": seq.Purples_7,
                "Purples_8": seq.Purples_8,
                "Purples_9": seq.Purples_9,
                "RdPu_3": seq.RdPu_3,
                "RdPu_4": seq.RdPu_4,
                "RdPu_5": seq.RdPu_5,
                "RdPu_6": seq.RdPu_6,
                "RdPu_7": seq.RdPu_7,
                "RdPu_8": seq.RdPu_8,
                "RdPu_9": seq.RdPu_9,
                "Reds_3": seq.Reds_3,
                "Reds_4": seq.Reds_4,
                "Reds_5": seq.Reds_5,
                "Reds_6": seq.Reds_6,
                "Reds_7": seq.Reds_7,
                "Reds_8": seq.Reds_8,
                "Reds_9": seq.Reds_9,
                "YlGnBu_3": seq.YlGnBu_3,
                "YlGnBu_4": seq.YlGnBu_4,
                "YlGnBu_5": seq.YlGnBu_5,
                "YlGnBu_6": seq.YlGnBu_6,
                "YlGnBu_7": seq.YlGnBu_7,
                "YlGnBu_8": seq.YlGnBu_8,
                "YlGnBu_9": seq.YlGnBu_9,
                "YlGn_3": seq.YlGn_3,
                "YlGn_4": seq.YlGn_4,
                "YlGn_5": seq.YlGn_5,
                "YlGn_6": seq.YlGn_6,
                "YlGn_7": seq.YlGn_7,
                "YlGn_8": seq.YlGn_8,
                "YlGn_9": seq.YlGn_9,
                "YlOrBr_3": seq.YlOrBr_3,
                "YlOrBr_4": seq.YlOrBr_4,
                "YlOrBr_5": seq.YlOrBr_5,
                "YlOrBr_6": seq.YlOrBr_6,
                "YlOrBr_7": seq.YlOrBr_7,
                "YlOrBr_8": seq.YlOrBr_8,
                "YlOrBr_9": seq.YlOrBr_9,
                "YlOrRd_3": seq.YlOrRd_3,
                "YlOrRd_4": seq.YlOrRd_4,
                "YlOrRd_5": seq.YlOrRd_5,
                "YlOrRd_6": seq.YlOrRd_6,
                "YlOrRd_7": seq.YlOrRd_7,
                "YlOrRd_8": seq.YlOrRd_8,
                "YlOrRd_9": seq.YlOrRd_9
            },
            "cubehelix": {
                "classic_16": cubhx.classic_16,
                "cubehelix1_16": cubhx.cubehelix1_16,
                "cubehelix2_16": cubhx.cubehelix2_16,
                "cubehelix3_16": cubhx.cubehelix3_16,
                "jim_special_16": cubhx.jim_special_16,
                "perceptual_rainbow_16": cubhx.perceptual_rainbow_16,
                "purple_16": cubhx.purple_16,
                "red_16": cubhx.red_16
            },
            "tableau": {
                "BlueRed_12": tb.BlueRed_12,
                "BlueRed_6": tb.BlueRed_6,
                "ColorBlind_10": tb.ColorBlind_10,
                "Gray_5": tb.Gray_5,
                "GreenOrange_12": tb.GreenOrange_12,
                "GreenOrange_6": tb.GreenOrange_6,
                "PurpleGray_12": tb.PurpleGray_12,
                "PurpleGray_6": tb.PurpleGray_6,
                "TableauLight_10": tb.TableauLight_10,
                "TableauMedium_10": tb.TableauMedium_10,
                "Tableau_10": tb.Tableau_10,
                "Tableau_20": tb.Tableau_20,
                "TrafficLight_9": tb.TrafficLight_9
            },
            "wesanderson": {
                "Aquatic1_5": wes.Aquatic1_5,
                "Aquatic2_5": wes.Aquatic2_5,
                "Aquatic3_5": wes.Aquatic3_5,
                "Cavalcanti_5": wes.Cavalcanti_5,
                "Chevalier_4": wes.Chevalier_4,
                "Darjeeling1_4": wes.Darjeeling1_4,
                "Darjeeling2_5": wes.Darjeeling2_5,
                "Darjeeling3_5": wes.Darjeeling3_5,
                "FantasticFox1_5": wes.FantasticFox1_5,
                "FantasticFox2_5": wes.FantasticFox2_5,
                "GrandBudapest1_4": wes.GrandBudapest1_4,
                "GrandBudapest2_4": wes.GrandBudapest2_4,
                "GrandBudapest3_6": wes.GrandBudapest3_6,
                "GrandBudapest4_5": wes.GrandBudapest4_5,
                "GrandBudapest5_5": wes.GrandBudapest5_5,
                "Margot1_5": wes.Margot1_5,
                "Margot2_4": wes.Margot2_4,
                "Margot3_4": wes.Margot3_4,
                "Mendl_4": wes.Mendl_4,
                "Moonrise1_5": wes.Moonrise1_5,
                "Moonrise2_4": wes.Moonrise2_4,
                "Moonrise3_4": wes.Moonrise3_4,
                "Moonrise4_5": wes.Moonrise4_5,
                "Moonrise5_6": wes.Moonrise5_6,
                "Moonrise6_5": wes.Moonrise6_5,
                "Moonrise7_5": wes.Moonrise7_5,
                "Royal1_4": wes.Royal1_4,
                "Royal2_5": wes.Royal2_5,
                "Royal3_5": wes.Royal3_5,
                "Zissou_5": wes.Zissou_5   
            }}

        #Select one palette
        self.palette = palettes[pal][col]

        #Name of the palette
        self.name = self.palette.name

        #One of the "diverging", "qualitative", "sequential"
        self.type = self.palette.type

        #Number
        self.number = self.palette.number

        #Colors RGB(0-255)
        self.colors = self.palette.colors

        #Hex colors ("#A912F4")
        self.hex_colors = self.palette.hex_colors

        #mpl_colors as (0-1) python default
        self.mpl_colors = self.palette.mpl_colors

        #A continous interpolated matplotlib
        self.mpl_colormap = self.palette.mpl_colormap

        #Methods for palette
        #Gets matplotlib colormap and pass custum keywork arguments to
        #LineaSegmentedColormap.from_list
        self.get_mpl_colormap = self.palette.get_mpl_colormap

        #Show the defined colors of the palette in the IPython Notebook. 
        #Requires ipythonblocks to be installed.
        self.show_as_blocks = self.palette.show_as_blocks

        #Show the defined colors of the palette in the IPython Notebook. 
        #Requires matplotlib to be installed.
        self.show_discrete_image = self.palette.show_discrete_image

        #Show the continuous, interpolated palette in the IPython Notebook. 
        #Requires matplotlib to be installed.
        self.show_continuous_image = self.palette.show_continuous_image

        #Save an image of the defined colors of palette to a file. 
        #Requires matplotlib to be installed.
        self.save_discrete_image = self.palette.save_discrete_image

        #Save an image of the continuous, interpolated palette to a file. 
        #Requires matplotlib to be installed.
        self.save_continuous_image = self.palette.save_continuous_image

    def chompColors(self,start,end):
        """
        Chomps colors from the original palette.

        Usefull when just want an specific part of the palette.

        If you want to chomp it 
        :Arguments:
            :type start: int
            :param start: Possition to start the subsetting

            :type start: end
            :param start: Possition to end the subsetting
        """
        # Subsetting colors as python default
        self.mpl_colors = self.mpl_colors[int(start):int(end)]

        # Re calculate new colormap based on the colors.
        self.mpl_colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
                            colors=self.mpl_colors, 
                            name='subseted')

    def getColorsCmapPalette(self,elements):
        """ 
        Gets a list  of colors for a given list of elements

        :Arguments:
            :type elements: list
            :param elements: list of elements to get colors from.

        :Returns:
            :rtype design: list
            :return design: list of colors.

        """

        #Creates a np array of the list rangin form 0-1
        colPosition = np.arange(0,1,1.0/len(elements))

        #Get an array of positions in the colormap 
        colPosition = np.array([x+(1.0/(len(elements)*2)) for x in colPosition])

        #Get list of colors out of the positions
        colorList = self.mpl_colormap(colPosition)

        #Return colorList
        return colorList

    def getColors(self,design,groups):
        """ 
        Get colors based on a desing file

        :Arguments:
            :type design: pandas.DataFrame
            :param design: A design file.

            :type groups: string.
            :param groups: the name of the column on design file that contains the groups

        :Returns:
            :rtype design: pandas.dataFrame
            :return design: Copy of the original design file with a new column with the colors

            :rtype ugColors: dictionary
            :return ugColors: dictionary with unique groups and its respective color
                                (useful for legeneds)

            :rtype combName: string
            :return combName: Column on the design that contains the combinations
        """

        #Getting the columns we are interested in dropping missing data columns
        if len(groups):
            self.design = design.loc[:,groups].dropna(axis=0)
        else:
            self.design = pd.DataFrame(design.index.values,index=design.index,columns=["samples"])
            self.design.index.name = "sampleID"

        #Getting groups that exists in the design file after dropping
        groups = self.design.columns.values

        #Creating combinations
        self.combName = "_".join(groups)
        
        #Creating combinations
        self.design.loc[:,self.combName] = self.design.apply(lambda x: "_".join(map(str,
                                    x[groups].values)),axis=1)
        #Getting uniq combinations
        uGroups = list(set(self.design[self.combName].values))

        #Remove the nan in the selected group column 

        #Get colours
        if self.combName == "samples":
            colorsInPalette = self.getColorByIndex(dataFrame=self.design,color_index=0)
        elif len(uGroups) > self.palette.number:
            colorsInPalette = self.getColorsCmapPalette(uGroups) 
        else:
            colorsInPalette = self.palette.mpl_colors[:len(uGroups)]
        
        #Get colors for each group
        self.ugColors = {group:color for group,color in zip(uGroups,colorsInPalette)}
        
        #Creating color for each combination
        self.design.loc[:,"colors"] = self.design[self.combName].apply(lambda x: self.ugColors[x])

        # Treat columns "samples" as its own group 
        if "samples" in self.design.columns:
            self.design.drop("samples",axis=1,inplace=True)
            self.design.loc[:,"samples"] = ["samples"] * len(self.design.index)

        #if only one group then use samples as ugroup
        if "samples" in self.design.columns:
            self.ugColors={"samples":list(set(colorsInPalette))[0]}

        #Creates a list of colors from the given dictionary works for groups and not  groups
        if "samples" in self.design.columns:
            self.list_colors = [self.ugColors["samples"]] * len(self.design.index)
        else:
            self.list_colors = [self.ugColors[group] for group in self.design[self.combName].values]

        #Returning design
        return self.design,self.ugColors,self.combName
        
    def getColorsByGroup(self,design,group,uGroup):
        """ 
        Gets a list of colors for groups

        :Arguments:
            :type design: pandas.DataFrame
            :param design: A design file.

            :type group: string.
            :param group: Name of the column on the design that contains the groups

            :type uGroup: list.
            :param uGroup: List of the unique groups on the design file

        :Returns:
            :rtype ugColors: list
            :return ugColors: list of colors (per group) for each one of the indexes
                                on the design.
            :rtype combName: dictionary
            :return combName: Dictionary with group:color
        """
        #design file, name of col in des that has groups (treatment), unique Groups 
        
        # Calculates colors form a cmap if the number of groups is bigger
        # than the defaultnumber of colors on the palette.
        if len(uGroup) > self.palette.number:
            colorsInPalette = self.getColorsCmapPalette(uGroup) 
        else:
            colorsInPalette = self.palette.mpl_colors[:len(uGroup)]

        #Creates a dictionary of group:color
        ugColors = {grp:color for grp,color in zip(uGroup,colorsInPalette)}

        #Creates a list of colors from the given dictionary 
        colors = [ugColors[group] for val,group in zip(design.index.values,
                design[group].values)] 

        #Return list of colours and dictionary of colors
        return colors,ugColors

    def getColorByIndex(self,dataFrame,color_index=0):
        """
        This functiuon gets colors for each index of a given dataFrame
        
        :Arguments:
            :type dataFrame: pandas.DataFrame
            :param dataFrame: Any given dataframe

            :type color_index: int.
            :param color_index: color of a palette that you want to give to
                                every element on the dataframe.

        :Returns:
            :rtype colors: list
            :return colors: list of colors for each index (all elements in 
                            the have the same color)
        """
        #Get indexes
        indexes = dataFrame.index.values

        #Set the color that you want for everything
        colors = [self.palette.mpl_colors[color_index]] * len(indexes)

        #Return list of colors
        return colors
