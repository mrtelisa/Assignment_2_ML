from one_dim_lr import *

def mse_one_dim(dataset, x, y):

    w_curr = one_dim_lr(dataset)
    #print(w0, "\n")

    mse_1 = 0
    y_prev_1 = []

    for j in range(len(dataset)):
        y_prev_1.append(w_curr*x[j]) 
        #print(f"y_prev_{i+1}:", y_prev_1, "\n\n"
    
    for j in range(len(y)):
        mse_1 += (y_prev_1[j]-y[j])**2
    J = mse_1 / len(x)

    return J