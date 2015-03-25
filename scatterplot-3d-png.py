#!/usr/bin/env python
#Greg Von Kuster

import sys
import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def stop_err(msg):
    sys.stderr.write(msg)
    sys.exit()

def main():

    in_fname = sys.argv[1]
    out_fname = sys.argv[2]
    out_fname_png = out_fname + '.png'
    try:
        columns = int( sys.argv[3] ) - 1, int( sys.argv[4] ) - 1, int( sys.argv[5] ) - 1
    except:
        stop_err( "Columns not specified, your query does not contain a column of numerical data." )
    title = sys.argv[6]
    xlab = sys.argv[7]
    ylab = sys.argv[8]
    zlab = sys.argv[9]
    displaywindow = sys.argv[10]
    displaywindow = False

    matrix = []
    xs = []
    ys = []
    zs = []
    skipped_lines = 0
    first_invalid_line = 0
    invalid_value = ''
    invalid_column = 0
    i = 0
    for i, line in enumerate( file( in_fname ) ):
        valid = True
        line = line.rstrip( '\r\n' )
        if line and not line.startswith( '#' ): 
            row = []
            fields = line.split( "\t" )
            for column in columns:
                try:
                    val = fields[column]
                    if val.lower() == "na": 
                        row.append( float( "nan" ) )
                    else:
                        row.append( float( fields[column] ) )
                        if column == columns[0]:
                            xs.append( float( fields[column] ) )
                        if column == columns[1]:
                            ys.append( float( fields[column] ) )
                        if column == columns[2]:
                            zs.append( float( fields[column] ) )
                except:
                    valid = False
                    skipped_lines += 1
                    if not first_invalid_line:
                        first_invalid_line = i + 1
                        try:
                            invalid_value = fields[column]
                        except:
                            invalid_value = ''
                        invalid_column = column + 1
                    break
        else:
            valid = False
            skipped_lines += 1
            if not first_invalid_line:
                first_invalid_line = i+1

        if valid:
            matrix.append( row )

    if skipped_lines < i:
        try:
	    #Define Figure
            fig = plt.figure()
            ax = fig.add_subplot( 111, projection='3d' )
            ax.scatter(xs, ys, zs, c='r', marker='o')


	    #Set lables
            ax.set_xlabel( xlab )
            ax.set_ylabel( ylab )
            ax.set_zlabel( zlab )

	    #Write png
            savefig( out_fname, format='png' )

	    #Show plot for testing purposes.
#	    if displaywindow:
#                plt.show()

	    #Copy to data file
	    #assert not os.path.isabs(out_fname_png)
	    #dstdir =  os.path.join(dstroot, os.path.dirname(out_fname_png))

	    #os.makedirs(dstdir) # create all directories, raise an error if it already exists
	    #shutil.copy(out_fname_png, out_fname_png)
            
        except Exception, exc:
            stop_err( "%s" %str( exc ) )
    else:
        stop_err( "All values in both columns %s and %s are non-numeric or empty." % ( sys.argv[3], sys.argv[4], sys.argv[5] ) )

    print "Scatter plot on columns %s, %s, %s. " % ( sys.argv[3], sys.argv[4], sys.argv[5] )
    if skipped_lines > 0:
        print "Skipped %d lines starting with line #%d, value '%s' in column %d is not numeric." % ( skipped_lines, first_invalid_line, invalid_value, invalid_column )

if __name__ == "__main__":
    main()
