#!/usr/bin/env python
################################################################################
# SCRIPT: dropFlag_v2.py
# 
# AUTHOR: Miguel Ibarra Arellano (miguelib@ufl.edu)
# 
# DESCRIPTION: This script takes a Wide format file (wide), a flag file and a 
# design file (only for drop by column) and drops either rows or columns for a 
# given criteria. This criteria could be either numeric,string or a flag (1,0). 
#
# OUTPUT:
#       Drop by row:
#                   Wide file with just the dropped rows
#                   Wide file without the dropped rows
#       Drop by column:
#                   Wide file with just the dropped columns
#                   Wide file without the dropped columns
#
################################################################################

#Standard Libraries
import logging
import argparse
import copy
from argparse import RawDescriptionHelpFormatter
import re

#Add-on Libraries
import pandas as pd

#Local Libraries
from interface import wideToDesign
import logger as sl


"""Function to pull arguments"""
def getOptions():
	parser = argparse.ArgumentParser(description="Drops rows or columns given \
									an specific cut value and condition")
	
	required = parser.add_argument_group(title='Required Input', 
										description='Requiered input to the \
										program.')
	required.add_argument('-i',"--input",dest="input", action="store",
						required=True,help="Input dataset in wide format.")
	required.add_argument('-d',"--design",dest="designFile", action="store",
						required=True, help="Design file.")
	required.add_argument('-f',"--flags",dest="flagFile", action="store",
						required=True,help="Flag file.")
	required.add_argument('-id',"--ID",dest="uniqID",action="store",
						required=True,help="Name of the column with unique \
						identifiers.")
	required.add_argument('-fd',"--flagDrop",dest="flagDrop",action='store',
						required=True, help="Name of the flag/field you want to\
						access.")

	output = parser.add_argument_group(title='Output', description='Output of \
										the script.')
	output.add_argument('-ow',"--outWide",dest="outWide",action="store",required=True,
						help="Output file without the Drops.")
	output.add_argument('-of',"--outFlags",dest="outFlags",action="store",
						required=True,help="Output file for Drops.")

	"""
	exclusive = parser.add_mutually_exclusive_group(required=True)
	exclusive.add_argument("-r","--row",dest="dropRow",action='store_true',
							help="Drop rows.")
	exclusive.add_argument("-c","--column",dest="dropColumn",action='store_true',
							help="Drop column.")
	"""

	optional = parser.add_argument_group(title="Optional Input", description="\
										Optional Input to the program.")
	required.add_argument('-g',"--group",dest="group",action="store",
						required=False,help="Group/treatment identifier in \
						design file.")
	optional.add_argument('-val',"--value",dest="value",action='store',
						required=False, default="1",help="Cut Value")
	optional.add_argument('-con',"--condition",dest="condition",action='store',
						required=False, default="0",help="Condition for the cut\
						where 0=Equal to, 1=Greater than and 2=less than.")

	args = parser.parse_args()
	return(args);

def dropRows(df_wide, df_flags,cut_value, condition, args):
	""" 
	Drop rows in a wide file based on its flag file and the specified flag 
	values to keep.

	:Arguments:
		:type df_wide: pandas.DataFrame
		:param df: A data frame in wide format

		:type df_flags: pandas.DataFrame
		:param df: A data frame of flag values corresponding to the wide file

		:type cut_value: string
		:param args: Cut Value for evaluation

		:type condition: string
		:param args: Condition to evaluate

		:type args: argparse.ArgumentParser
		:param args: Command line arguments.

	:Returns:
		:rtype: pandas.DataFrame
		:returns: Updates the wide DataFrame with dropped rows and writes to a
			TSV.
		:rtype: pandas.DataFrame
		:returns: Fron wide DataFrame Dropped rows and writes to a TSV.
	"""
	#Dropping flags from flag files, first asks for the type of value, then asks
	# for the diferent type of conditions new conditios can be added here

	if re.match('^[0-9]',cut_value):
		cut_value = float(cut_value)
		if condition == '>':
			df_filtered =  df_flags[df_flags[args.flagDrop]<cut_value]
		elif condition == '<':
			df_filtered =  df_flags[df_flags[args.flagDrop]>cut_value]
		elif condition == '==':
			df_filtered =  df_flags[df_flags[args.flagDrop]!=cut_value]
		else:
			logger.error(u'The {0} is not supported by the program, please use <,== or >'.format(condition))
			quit()
	else:
		cut_value = str(cut_value)
		if condition == '==':
			df_filtered =  df_flags[df_flags[args.flagDrop]!=cut_value]
		else:
			logger.error(u'The {0} conditional is not supported for string flags, please use =='.format(condition))
			quit()

	#Create a mask over the original data to determinate what to delete
	mask = df_wide.index.isin(df_filtered.index)

	#Create a mask over the original flags to determinate what to delete
	mask_flags = df_flags.index.isin(df_filtered.index)

	# Use mask to drop values form original data
	df_wide_keeped = df_wide[mask]
	#df_wide_dropped = df_wide[~mask]

	# Use mas to drop values out of original flags
	df_flags_keeped = df_flags[mask_flags]
	#df_flags_dropped = df_flags[~mask_flags]

	#Export wide
	df_wide_keeped.to_csv(args.outWide, sep='\t')
	#df_wide_dropped.to_csv(args.outputDrops, sep='\t')

	#Export flags
	df_flags_keeped.to_csv(args.outFlags, sep='\t')
	#df_flags_dropped.to_csv(args.outputDrops, sep='\t')

def dropColumns(df_wide, df_flags,cut_value, condition, args):
	""" 
	Drop columns in a wide file based on its flag file and the specified flag 
	values to keep.

	:Arguments:
		:type df_wide: pandas.DataFrame
		:param df: A data frame in wide format

		:type df_flags: pandas.DataFrame
		:param df: A data frame of flag values corresponding to the wide file

		:type cut_value: string
		:param args: Cut Value for evaluation

		:type condition: string
		:param args: Condition to evaluate

		:type args: argparse.ArgumentParser
		:param args: Command line arguments.

	:Returns:
		:rtype: pandas.DataFrame
		:returns: Updates the wide DataFrame with dropped columns and writes to 
					a TSV.
		:rtype: pandas.DataFrame
		:returns: Fron wide DataFrame Dropped columns and writes to a TSV.
	"""
	#Getting list of filtered columns from flag files
	if re.match('^[0-9]',cut_value):
		cut_value = float(cut_value)
		if condition == '>':
			samples_to_drop = df_flags.index[df_flags[args.flagDrop]<cut_value]
		elif condition == '<':
			samples_to_drop = df_flags.index[df_flags[args.flagDrop]>cut_value]
		elif condition == '==':
			samples_to_drop = df_flags.index[df_flags[args.flagDrop]!=cut_value]
		else:
			logger.error(u'The {0} is not supported by the program, please use <,== or >'.format(condition))
			quit()
	else:
		cut_value = str(cut_value)
		if condition == '==':
			samples_to_drop = df_flags.index[df_flags[args.flagDrop]!=cut_value]
		else:
			logger.error(u'The {0} conditional is not supported for string flags, please use =='.format(condition))
			quit()

	dropped_flags = df_flags.T[samples_to_drop].T

	#Output 
	df_wide.to_csv(args.outWide, columns=samples_to_drop, sep='\t')
	dropped_flags.to_csv(args.outFlags, sep='\t')


def main():
	#Gettign arguments from parser
	args = getOptions()

	#Stablishing logger
	logger = logging.getLogger()
	sl.setLogger(logger)

	#Change condition
	if args.condition == "0":
		args.condition="=="
	elif args.condition == "1":
		args.condition=">"
	elif args.condition == "2":
		args.condition="<"

	#Starting script
	logger.info(u'Importing data with following parameters: \
		\n\tWide: {0}\
		\n\tFlags: {1}\
		\n\tDesign: {2}\
		\n\tID: {3}\
		\n\tgroup: {4}\
		\n\toutput: {5}\
		\n\tVariable: {6}\
		\n\tCondition: {7}\
		\n\tValue: {8}'.format(args.input,args.flagFile,args.designFile,
							args.uniqID,args.group,args.outWide,args.outFlags,
							args.condition,args.value))

	# Execute wideToDesign to make all data uniform
	formatted_data = wideToDesign(wide=args.input, design=args.designFile, 
								uniqID=args.uniqID, group=args.group)    

	# Convert flag file to DataFrame
	df_flags = pd.DataFrame.from_csv(args.flagFile, sep='\t')
	
	# Drop wither rows or columns
	if df_flags.index.name=="sampleID":
		logger.info("Runing drop flags by Column")
		dropColumns(df_wide=formatted_data.wide, df_flags=df_flags, 
					cut_value=args.value, condition=args.condition, args=args)
	
	else:
		logger.info("Runing drop flags by Row")
		dropRows(df_wide=formatted_data.wide, df_flags=df_flags, 
				cut_value=args.value, condition=args.condition, args=args)

	# Finishing script
	logger.info("Script complete.")

if __name__ == '__main__':
	main()
	