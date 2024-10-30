import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from one_dim_lr import *
from one_dim_lr_off import *
from multi_dim_lr import *
from mse_one_dim import *

############################################################################################################### TASK 1
# TASK 1: Loading the data

# from turkish-se-SP500vsMSCI.csv
dataEx = pd.read_csv("turkish-se-SP500vsMSCI.csv")
coordinates = []
coordinates = dataEx.columns

rowEx = dataEx[coordinates[0]].tolist()
colEx = dataEx[coordinates[1]].tolist()

# from mtcarsdata-4features.csv
dataCar = pd.read_csv("mtcarsdata-4features.csv")
coordinates1 = []
coordinates1 = dataCar.columns

mpg = dataCar[coordinates1[1]].tolist()
disp = dataCar[coordinates1[2]].tolist()
hp = dataCar[coordinates1[3]].tolist()
weight = dataCar[coordinates1[4]].tolist()

############################################################################################################### TASK 2
# TASK 2

# 1-D Linear regression without interception with all the data from turkish-se-SP500vsMSCI.csv

w_all = one_dim_lr(dataEx)

# Creating and plotting the graph
plt.scatter(rowEx, colEx, marker = "x")
x_vals = np. linspace(min(rowEx), max(rowEx), 100)

y_vals = w_all * x_vals
plt.plot(x_vals, y_vals, color="red")

plt.xlabel("Standards and Poor's 500 return index")
plt.ylabel("MSCI Europe index")
plt.show()

###############################################################################################################
# 1-D Linear regression without interception with 10% of the data from turkish-se-SP500vsMSCI.csv

w0 =[]
for i in range(5):
    randomSubset = np.random.permutation(len(rowEx))[:round(len(rowEx)*0.1)]
    datasetEx = dataEx.iloc[randomSubset]
    w0.append(one_dim_lr(datasetEx))
    #print(w0, "\n")

# Creating and plotting the graph
plt.scatter(rowEx, colEx, marker = "x")
x_vals = np. linspace(min(rowEx), max(rowEx), 100)

for i, coeff in enumerate(w0):
    y_vals = coeff * x_vals
    plt.plot(x_vals, y_vals, label = f"w_{i+1} = {coeff:.3f}")

plt.legend()
plt.xlabel("Standards and Poor's 500 return index")
plt.ylabel("MSCI Europe index")
plt.show()

###############################################################################################################
# 1-D Linear regression with interception using mpg and weight from mtcarsdata-4features.csv

w0Car, w1Car = one_dim_lr_off(pd.DataFrame({'x': weight, 'y': mpg}))

# Creating and plotting the graph
plt.scatter(weight, mpg, marker='x') 
x_vals = np.linspace(min(weight), max(weight), 100)
y_vals = w0Car + (w1Car * x_vals)  

plt.plot(x_vals, y_vals, color = "Red")
plt.xlabel("Car weight (lbs/1000)")
plt.ylabel("mpg")
plt.show()

###############################################################################################################
# Multi-dim Linear regression with interception using all data from mtcarsdata-4features.csv

dataMulti = pd.DataFrame({'disp': disp, 'hp': hp, 'weight': weight, "mpg": mpg})

w, predictions = multi_dim_lr(dataMulti)
error = [mpg[i] - predictions[i] for i in range(len(mpg))]
comparison = np.column_stack((mpg, predictions,error))
np.set_printoptions(precision=3, suppress=True)
print("\n\nComparison between actual mpg, mpg prediction and value of the error:\n", comparison)


############################################################################################################### TASK 3
# TASK 3

rep = 10

msetrain1 = []
msetest1 = []
msetrain3 = []
msetest3 = []
msetrain4 = []
msetest4 = []

for i in range(rep):
    # initializing variables
    objTrainig1 = 0
    objTest1 = 0
    objTrainig3 = 0
    objTest3 = 0
    objTrainig4 = 0
    objTest4 = 0

    ###################### Dividing training set and test set for both the files
    # training_set
    trainingSetIndicesEx = np.random.permutation(len(rowEx))[:round(len(rowEx)*0.15)]
    trainingSetEx = dataEx.iloc[trainingSetIndicesEx]
    trainingSetIndicesCar = np.random.permutation(len(mpg))[:round(len(mpg)*0.15)]
    trainingSetCar = dataCar.iloc[trainingSetIndicesCar]

    # test_set
    testSetIndicesEx = np.setdiff1d(np.arange(len(rowEx)), trainingSetIndicesEx)
    testSetEx = dataEx.iloc[testSetIndicesEx]
    testSetIndicesCar = np.setdiff1d(np.arange(len(mpg)), trainingSetIndicesCar)
    testSetCar = dataCar.iloc[testSetIndicesCar]
    
    ###################### Loading data
    # loading training_set ex
    coordinates_trEx = []
    coordinates_trEx = trainingSetEx.columns
    trainingSetRowEx = trainingSetEx[coordinates_trEx[0]].tolist()
    trainingSetColEx = trainingSetEx[coordinates_trEx[1]].tolist()

    # loading test_set ex
    coordinates_tsEx = []
    coordinates_tsEx = testSetEx.columns
    testSetRowEx = testSetEx[coordinates_tsEx[0]].tolist()
    testSetColEx = testSetEx[coordinates_tsEx[1]].tolist()

    # loading training_set car
    coordinates_trCar = []
    coordinates_trCar = trainingSetCar.columns
    trainingSetMpg = trainingSetCar[coordinates_trCar[1]].tolist()
    trainingSetDisp = trainingSetCar[coordinates_trCar[2]].tolist()
    trainingSetHp = trainingSetCar[coordinates_trCar[3]].tolist()
    trainingSetWeight = trainingSetCar[coordinates_trCar[4]].tolist()

    # loading test_set car
    coordinates_tsCar = []
    coordinates_tsCar = testSetCar.columns
    testSetMpg = testSetCar[coordinates_tsCar[1]].tolist()
    testSetDisp = testSetCar[coordinates_tsCar[2]].tolist()
    testSetHp = testSetCar[coordinates_tsCar[3]].tolist()
    testSetWeight = testSetCar[coordinates_tsCar[4]].tolist()   

    ###################### Computing the MSE for ex data
    w = one_dim_lr(trainingSetEx)

    # training_set ex
    for i in range(len(trainingSetRowEx)):
        trainingPred1 = w * trainingSetRowEx[i]
        trainingError1 = (trainingSetColEx[i] - trainingPred1)**2
        objTrainig1 += trainingError1

    msetrain1.append(objTrainig1/len(trainingSetRowEx))

    # test_set ex
    for i in range(len(testSetRowEx)):
        testPred1 = w * testSetRowEx[i]
        testError1 = (testSetColEx[i] - testPred1)**2
        objTest1 += testError1

    msetest1.append(objTest1/len(testSetRowEx))
    
    ###################### Computing the MSE for mpg and weight with interception
    w0TrainCar, w1TrainCar = one_dim_lr_off(pd.DataFrame({'x': trainingSetWeight, 'y': trainingSetMpg}))

    # training_set car    
    for i in range(len(trainingSetWeight)):
        trainingPred3 = (w1TrainCar * trainingSetWeight[i]) + w0TrainCar
        trainingError3 = (trainingSetMpg[i] - trainingPred3)**2
        objTrainig3 += trainingError3

    msetrain3.append(objTrainig3/len(trainingSetWeight))

    # test_set car
    for i in range(len(testSetWeight)):
        testPred3 = (w1TrainCar * testSetWeight[i]) + w0TrainCar
        testError3 = (testSetMpg[i] - testPred3)**2
        objTest3 += testError3

    msetest3.append(objTest3/len(testSetWeight))

    ###################### Computing the MSE for car data with interception

    # training_set car with interception
    xtrain = pd.DataFrame({'disp': trainingSetDisp, 'hp': trainingSetHp, 'weight': trainingSetWeight, 'mpg': trainingSetMpg })
    Wtrain,predTraining = multi_dim_lr(xtrain)
    for i in range(len(trainingSetMpg)):
        objTrainig4 += (trainingSetMpg[i] - predTraining[i])**2

    msetrain4.append(objTrainig4/2)

    # test_set car with interception
    xtest = pd.DataFrame({'disp': testSetDisp, 'hp': testSetHp, 'weight': testSetWeight, 'mpg': testSetMpg })
    Wtest,predTest = multi_dim_lr(xtest,Wtrain)
    for i in range(len(testSetMpg)):
        objTest4 += (testSetMpg[i] - predTest[i])**2
    msetest4.append(objTest4/2)

# Comparing the obtained results
  
comparison = np.column_stack((msetrain1, msetest1, msetrain3, msetest3, msetrain4, msetest4))

column_headers = ["MSE Train 1", "MSE Test 1", "MSE Train 3", "MSE Test 3", "MSE Train 4", "MSE Test 4"]
comparison_df = pd.DataFrame(comparison, columns=column_headers)

comparison_df["MSE Train 1"] = comparison_df["MSE Train 1"].map(lambda x: f"{x:.3e}")
comparison_df["MSE Test 1"] = comparison_df["MSE Test 1"].map(lambda x: f"{x:.3e}")
comparison_df["MSE Train 3"] = comparison_df["MSE Train 3"].round(3)
comparison_df["MSE Test 3"] = comparison_df["MSE Test 3"].round(3)
comparison_df["MSE Train 4"] = comparison_df["MSE Train 4"].round(3)
comparison_df["MSE Test 4"] = comparison_df["MSE Test 4"].round(3)

print("\n\n", comparison_df.to_string(index=False))
