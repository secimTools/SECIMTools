#!/bin/bash - 
#===============================================================================
# 
#   DESCRIPTION: Clone the galaxy distribution and checkout the branch that the
#   HPC is using.
# 
#===============================================================================

# Create devel location
if [ ! -e $HOME/devel ]; then mkdir -p $HOME/devel; fi

# Clone galaxy
hg clone https://bitbucket.org/galaxy/galaxy-dist $HOME/devel/

# pull the HPC version
cd $HOME/devel/galaxy-dist
hg pull && hg update release_2014.06.02
