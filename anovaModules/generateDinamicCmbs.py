def generateDinamicCmbs(factors,globList,acum=False):
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
