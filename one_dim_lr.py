def one_dim_lr(dataset):
    columns = dataset.columns
    num, den = 0, 0
    x = dataset[columns[0]].tolist()
    y = dataset[columns[1]].tolist()   
        #print(dataset, "\n\n")
    for i in range(len(dataset)):
        num += x[i]*y[i]
        den += x[i]*x[i]
    return num/den