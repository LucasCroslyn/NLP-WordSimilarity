def MSE(preds, golds):
    '''
    Calculates the mean squared error between a list of predictions and their true, gold values

    :param preds: A list of the predictions that can be easily formatted as a float.
    :param golds: A list of what the predictions should give in the best case scenario and can be easily formatted into floats.
    :return: Returns the mean squared error over the entire list of data.
    '''
    
    assert len(preds) == len(golds), "Different number of predictions and golds"
    curr_MSE = 0
    for i in range(len(preds)):
        curr_MSE += (float(preds[i])-float(golds[i]))**2
    curr_MSE = float(curr_MSE)/float(len(golds))
    return curr_MSE