import pickle as pk


def pickleDict(objDict, fname):
    """ Pickle a dictionary of "var name": values for debugging.

    Arguments:
        :param dict objDict: A dictionary where each key is the name of a
            variable and the value is the object itself.

        :param str fname: File name of the pickle.
    """
    with open(fname, 'wb') as FH:
        pk.dump(objDict, FH)


def unPickleDict(fname):
    """ Pickle a dictionary of "var name": values for debugging.

    Arguments:
        :param str fname: File name of the pickle.

    Returns:
        :rtype: dict
        :returns: A dictionary where each key is the name of a
            variable and the value is the object itself.

    """
    with open(fname, 'rb') as FH:
        objDict = pk.load(FH)

    return objDict
