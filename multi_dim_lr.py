import numpy as np 

def multi_dim_lr(dataset, Win = None):

    columns = []
    columns = dataset.columns

    col1 = [1] * len(dataset)
    x = dataset[columns[0]].tolist()
    y = dataset[columns[1]].tolist()
    z = dataset[columns[2]].tolist()

    t = dataset[columns[3]].tolist()

    X = np.column_stack((col1,x,y,z))


    if Win is None:
        den = np.linalg.pinv(((X.T) @ X))
        num = (X.T) @ t

        W = den @ num
        pred = X @ W

        return W, pred
    else:
        pred = X @ Win
        return Win, pred