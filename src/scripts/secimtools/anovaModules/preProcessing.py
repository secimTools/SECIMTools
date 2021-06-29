import pandas as pd


def preProcessing(factorNames, factorTypes, design):
    """
    Pre processing of the data by obtaining the factor name and types,
    generating the formulas and makes sure that the factors are present 
    on the design and data.

    :Arguments:
        :type factorNames: str
        :param factorNames: comma separated string with the factor(s) the 
                            the user wants to use to run ANOVA.

        :type factorTypes: str
        :param factorTypes: comma separated string with the factor(s) type(s),
                            they should match in order with the factorNames.

        :type design: pandas.DataFrame.
        :param design: design file.

    :Returns:
        :rtype preFormula: str
        :return preFormula: right part of the ANOVA model, everything afer the '~'
                            Y ~ C(categorical1)+C(categorical2)

        :rtype categorical: list
        :return categorical: Only categorical factors

        :rtype numerical: list
        :return numerical: Only numerical factors

        :rtype levels: list
        :return levels: Elements for each categorical factors
    """

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

    # iterating over factor and factorType
    categorical = list()
    numerical = list()

    # Identify whether a factor is categorical or numerical
    for fName, fType in list(zip(factorNames, factorTypes)):
        if fName not in designCols:
            logger.error("'{}' is not located in your design file".format(factor))
        if fType == "C" or fType == "c":
            categorical.append(fName)
        elif fType == "N" or fType == "n":
            numerical.append(fName)
        else:
            logger.error(
                "'{0}' is not a Valid Flag, use a valid flag to specify"
                "Categorical(C|c) or Numerical (N|n).".format(fType)
            )

    # If nan found then replace it with 0
    design.fillna("___", inplace=True)

    # If categorical treat values as str if numerical as float
    for cat in categorical:
        design[cat] = design[cat].apply(str)

    # Get list of unique levels
    lvlsNams = [
        [sorted(list(set(design[category].tolist()))), category]
        for category in categorical
    ]

    # Sort levels by number of elements on levels
    lvlsNams = sorted(lvlsNams, key=lambda x: len(x[0]), reverse=True)
    lvlsNams = list(zip(*lvlsNams))

    # Geting sorted levels and names
    levels = list(lvlsNams[0])
    categorical = list(lvlsNams[1])

    preFormula = ["C({0})".format(cat) for cat in categorical]
    preFormula += numerical
    preFormula = "+".join(preFormula)
    return preFormula, categorical, numerical, levels, design
