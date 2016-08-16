import pandas as pd

def preProcessing(factorNames, factorTypes, design):
    # Split by ',' the names and the types of the factors
    factorNames = factorNames.split(",")
    factorTypes = factorTypes.split(",")
    
    # Get the list of columns on the design file
    designCols = design.columns.tolist()
    
    # Check len of the factors
    # I the number of elements on names foesnt match with
    # the types will rise an error
    if len(factorNames) != len(factorTypes):
        logger.error("Length of Factors doesnt match FactorType")
        
    #iterating over factor and factorType
    categorical = list()
    numerical   = list()
    
    # Identify wheter a factor is categorical or numerical
    for fName,fType in zip(factorNames,factorTypes):
        if fName not in designCols:
            logger.error("'{}' is not located in your design file".format(factor))
        if fType =="C" or fType=="c":
            categorical.append(fName)
        elif fType =="N" or fType=="n":
            numerical.append(fName)
        else:
            logger.error("'{0}' is not a Valid Flag, use a valid flag to specify \
                    Categorical(C|c) or Numerical (N|n)."\
                    .format(fType))
            
    # Get list of unique levels
    lvlsNams=[[sorted(list(set(design[category].tolist()))),category] for category \
            in categorical]
    
    # Sort levels by number of elements on levels
    lvlsNams = sorted(lvlsNams,key=lambda x:len(x[0]),reverse=True)
    lvlsNams =  zip(*lvlsNams)
    
    # Geting sorted levels and names
    levels = list(lvlsNams[0])
    categorical  = list(lvlsNams[1])
    
    # Once sorted create preformula
    preFormula = ["C({0})".format(cat) for  cat in categorical]

    # Adding numerical at the end
    preFormula+= numerical
    
    # Creating preFormula
    preFormula =  "+".join(preFormula)
          
    # Returning 
    return preFormula,categorical,numerical,levels