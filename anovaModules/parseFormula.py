def parseFormula(f,uid):
    # Remove blank 
    f = re.sub(" ","",f)
    
    # Remove not allowed characters
    f = re.sub("[!|@|#|$|%|^|&||-|_|=|[|]|{|}|\|/|||.|,|;]","",f)
    
    # Convert lowercase c to uppercase C
    f = re.sub("c\(","C(",f)
        

    # Keeping jsut the second part of the formula
    if "~"in f:
        preForm = f.split("~")[1]
    else:
        print "'~' is missing from your formula"
    
    # Replace all alowed  characther with \t to further tokenize
    f2 = re.sub("[~|*|+|:]","\t",f)
    f2 = re.sub("[C(|)]","",f2)
    
    # Getting Uniq tokens
    tokens = list(sorted(set(f2.split("\t"))))
    
    # Remove uid from tokens if fails raise error
    try:
        tokens.remove(uid)
    except:
        print "{0} is not located in your formula, writte your \
                formula again and make sure your first element match \
                your unique ID.".format(uid)    
        
    # Getting indexes for given token    
    tokDict = {tok:"{"+str(i)+"}"for i,tok in enumerate(tokens)}
    
    # Substitute values based on tokDict
    form = re.sub("|".join(tokens),lambda x: tokDict[x.group(0)],preForm)
    
    # Return parsed formula scheme without the metabolite and group 
    #of uniq tokens
    return "~"+form.format(*tokens), tokens