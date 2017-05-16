def generateDinamicCmbs(factors,globList,acum=False):
    """
    This function will generate dinamic combinations between elements
    from different factors. ie. give [x,y,z],[a,b,c] this toll will return
    [[x,a],[x,b],[x,c],[y,a],[y,b],[y,c],[z,a],[z,b],[z,c]].
    
    :Arguments:
        :type data_df: factors
        :param data_df: Initial list

        :type design: globList
        :param design: List were all the combinatiosn will be stored

        :type pdf: boolean
        :param pdf: acumulative list.
    """

    # If not factors left return None
    factor = factors.pop()
    
    # If factor found iterate over it
    for level in factor:
        if len(factors)>0:
            if acum:
                if acum[-1] in factor:
                    acum[-1] = level
                else:
                    acum=acum+[level]
            else:
                acum = [level]
            
            # If any factor left then recurse again
            generateDinamicCmbs(copy.deepcopy(factors),globList,acum)
            
        else:
            if acum:
                if acum[-1] in factor:
                    finalList[-1] = level
                else:
                    finalList=acum+[level]
            else:
                finalList = [level]
            
            globList.append(finalList)
