def one_dim_lr_off(dataset):
    
    columns = []
    columns = dataset.columns
    x = dataset[columns[0]].tolist()
    y = dataset[columns[1]].tolist()

    avgx = 0
    avgy = 0 

    for j in range(len(dataset)):
        avgx += x[j]
        avgy += y[j]

    x_m = avgx / len(dataset)
    y_m = avgy / len(dataset)

    num = 0
    den = 0
    for i in range(len(dataset)):
        num += (x[i] - x_m)*(y[i] - y_m)
        den += (x[i] - x_m)*(x[i] - x_m)
    w1 = num/den
    w0 = y_m - (w1*x_m)
    return w0, w1