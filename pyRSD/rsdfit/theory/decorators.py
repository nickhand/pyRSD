valid = ['systematic_free_P0']

def systematic_free_P0(*poles):
    """
    Return :math:`P_0 + 2/5 * P_2`.
    """
    if not len(poles) == 2:
        msg = "error calling 'systematic_free_P0' decorator; expected 2 input multipoles,"
        msg += " but received %d inputs" % len(poles)
        raise ValueError(msg)

    return poles[0] + 2./5 * poles[1]
