

def running_average(val, mean, count):
    """
    Based on Welford's method:
    M_1 = x_1
    M_k = M_{k-1} + (x_k - M_{k-1}) / (k)
    :param val:
    :param mean:
    :param count:
    :return:
    """
    return mean + (val - mean) / (count + 1)