import copy

def gimmeTheMissin (partiaList,fullList):
    """
    This function will return the missing value of an incomplete list compared
    to its complete list. ie ["y","z"] as partial list and ["x","y","z"] as full 
    will return ["x"]
    """
    missings= list()
    for pElem in partiaList:
        for fList in fullList:
            if pElem in fList:
                lis2 = copy.deepcopy(fList)
                for p2 in partiaList:
                    if  p2 in lis2:
                        lis2.remove(p2)
                missings.append(lis2.pop())
    return missings