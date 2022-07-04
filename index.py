
# function - read csv into dataframe
# make function to compute derivatives at different time slots
# function - save to csv
from cProfile import label
from black import out
import pandas as pd
import numpy as np
from yaml import load


def compute_derivative(data, interval):
    tmpData = np.array([])
    for i in range(0, len(data)-interval):
        tmpData = np.append(tmpData, (data[i] - data[i+interval])/interval)
    return tmpData

def compute_avg_intervals(data, interval):
    i = 0
    tmpData = np.array([])
    tmpSum = 0
    for d in data:
        tmpSum += d
        i +=1
        if interval == i:
            tmpData = np.append(tmpData, tmpSum/interval)
            i = 0
            tmpSum = 0
    return tmpData

# if d is zero???
def compute_avg_intervals_sum(data, interval):
    i = 0
    tmpData = np.array([])
    previous = 0.0
    tmpSum = 0.0
    for d in data:
        if(float(d) == 0.0):
            tmpSum += previous
        else:
            tmpSum += float(d)
            previous = float(d)
        i +=1
        if interval == i:
            i = 0
            tmpData = np.append(tmpData, tmpSum/interval)
    return tmpData
def descriptorModel(test):
    y = POgiven()
    X = descriptors()
    y= y[~np.isnan(X).any(axis=1)]
    X= X[~np.isnan(X).any(axis=1)]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)#, random_state=0)
    if test:
        return Ridge().fit(X_train, y_train), X_test, y_test 
    else:
        return Ridge().fit(X_train, y_train)

from sklearn.model_selection import train_test_split
def POgraph(iterations):
    y = POgiven()
    plotX = []
    plotY = []
    for j in range(iterations):
        for i in range(1,100): #(2,100):
            X  =  PO(i) # VO2(i)np.concatenate(POplotted(i), CadenecePlotted(i))
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=0)
            reg = Ridge().fit(X_train, y_train)
            plotX.append(i+14)
            plotY.append(reg.score(X_test, y_test))
    plt.scatter(plotX,plotY, c='black')

    a, b = np.polyfit(plotX, plotY, 1)
    plt.plot(np.array(plotX), a*np.array(plotX)+b, color='steelblue', linestyle='--', linewidth=4)
    plt.xlabel('Time used')
    plt.ylabel('Score')
    plt.title('Ridge regression - PO generated feature')
    plt.show()

def VO2graph():
    y = POgiven()
    for i in range(24,100): #(2,100):
        X  =  VO2(i) #np.concatenate(POplotted(i), CadenecePlotted(i))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=0)
        reg = LinearRegression().fit(X_train, y_train)
        plt.scatter(i+2,reg.score(X_test, y_test), c='black')
    plt.xlabel('Time used')
    plt.ylabel('Score')
    plt.title('Linear regression - VO2')
    plt.show()


def Cgraph():
    y = POgiven()
    for i in range(1,100): #(2,100):
        X  =  Cadenece(i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)#, random_state=0)
        reg = Ridge().fit(X_train, y_train)
        plt.scatter(i+5,reg.score(X_test, y_test), c='black')
    plt.xlabel('Time used')
    plt.ylabel('Score')
    plt.title('Ridge regression - Cadence')
    plt.show()

def Descriptorgraph(iterations):
    for i in range(iterations):
        reg, X_test,y_test = descriptorModel(True)
        plt.scatter(i,reg.score(X_test, y_test), c='black')
    
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.title('Descriptor models')
    plt.show()

def POgiven():
    df = pd.read_csv('PO.csv')
    outStuff = []
    import matplotlib.pyplot as plt
    for d in range(len(df.columns)):
        tmp = []
        # print(df.loc[d])
        # print(d)
        for dTmp in df.iloc[:,d]:
            try:
                tmp.append(float(dTmp))
            except ValueError:
                continue
        outStuff.append(sum(np.array(tmp[-30:]))/30)
    return np.array(outStuff)

def POplotted():
    df = pd.read_csv('PO.csv')
    outStuff = []
    import matplotlib.pyplot as plt
    for d in range(len(df.columns)):
        tmp = []
        for dTmp in df.iloc[:,d]:
            try:
                tmp.append(float(dTmp))
            except ValueError:
                continue
        ai = compute_avg_intervals_sum(tmp,1)
        der = compute_derivative(ai, 5)
        plt.plot(der,range(len(der)))
        # plt.plot(ai,range(len(ai)))
    plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
def descriptors():
    # dataset column is actually a userID column not a sampleID column
    # ignore when training - might be useful when finding power of each etc
    df = pd.read_csv('Descriptors.csv')
    groups = df['Dataset']
    df = df.drop('Dataset', axis=1)
    # df = df.drop('VO2peak (L.min-1)', axis=1)
    # df = df.drop('GET (L.min-1)', axis=1)
    # df = df.drop('GET (W)', axis=1)
    # df = df.drop('Peak Power Output (W) ', axis=1)
    df['SEX'] = [0 if s == 'M' else 1 for s in df['SEX']]
    # make male sex 0 and female 1

    return np.array(df), np.array(groups)

def VO2_plotted():
    df = pd.read_csv('VO2.csv')
    import matplotlib.pyplot as plt

    for d in range(len(df.columns)):
        tmp = []
        for dTmp in df.iloc[:,d]:
            try:
                tmp.append(float(dTmp))
            except ValueError:
                continue
        if(sum(np.array(tmp)) >0):
            der = tmp
            ai = compute_avg_intervals_sum(tmp,2)
            der = compute_derivative(ai, 2)
        plt.plot(ai,range(len(ai)))
    plt.show()


def PO(amount):
    df = pd.read_csv('PO.csv')
    outStuff = []
    peaks = []
    import matplotlib.pyplot as plt
    for d in range(len(df.columns)):
        tmp = []
        for dTmp in df.iloc[:,d]:
            try:
                tmp.append(float(dTmp))
            except ValueError:
                continue
        ai = compute_avg_intervals_sum(tmp,1)
        der = compute_derivative(ai, 5)
        outStuff.append(der[0:amount])
        peaks.append([np.max(np.array(tmp))])
    return np.array(outStuff), np.array(peaks)

def Cadenece(amount):
    df = pd.read_csv('Cadence.csv')
    outStuff = []
    peaks = []
    import matplotlib.pyplot as plt
    for d in range(len(df.columns)):
        tmp = []
        for dTmp in df.iloc[:,d]:
            try:
                tmp.append(float(dTmp))
            except ValueError:
                continue
        ai = compute_avg_intervals_sum(tmp,1)
        der = compute_derivative(ai, 5)
        outStuff.append(der[0:amount])

        peaks.append([np.max(np.array(tmp))])
    return np.nan_to_num(np.array(outStuff)),np.nan_to_num( np.array(peaks))

def VO2(amount):
    df = pd.read_csv('VO2.csv')
    import matplotlib.pyplot as plt

    outStuff = []
    peaks = []
    for d in range(len(df.columns)):
            tmp = []
            for dTmp in df.iloc[:,d]:
                try:
                    tmp.append(float(dTmp))
                except ValueError:
                    continue
            if(sum(np.array(tmp)) >0):
                der = compute_derivative(tmp[0:amount],1)
                der = compute_derivative(der,2)
            outStuff.append(der)

            peaks.append([np.max(np.array(tmp))])
    return np.array(outStuff), np.array(peaks)



# Descriptorgraph(100)
# POgraph(1)
# VO2graph()
# Cgraph()
# POplotted()
# CadenecePlotted()
# VO2_plotted()

def saveModel(model, name):
    from joblib import dump
    dump(model, name+'.joblib')
def loadModel(loc):
    from joblib import load
    return load(loc+'.joblib')
def binValues(vals, binSize):
    b = np.digitize(vals, range(110,380,binSize))
    b = 110 + binSize/2 + (b-1)*binSize
    return b
def wPrime():
    from scipy.integrate import simps
    df = pd.read_csv('PO.csv')
    power = np.array([])
    for d in range(len(df.columns)):
        tmp = np.array([])
        for dTmp in df.iloc[:,d]:

            try:
                tmp = np.concatenate((tmp,np.array([float(dTmp)])))
            except ValueError:
                tmp = np.concatenate((tmp,np.array([0])))
        power = np.concatenate((power, tmp), axis = 0)
    power = power.reshape((211,180))
    cp = POgiven()
    w = np.array([p -c for c,p in zip(cp, power)])
    scores = []
    for W,c in zip(w,cp):
        tmp = [0]
        score = 0
        for p in W:
            if p >0:
                tmp.append(p)
            else:
                tmp.append(0)
                score += simps(np.array(tmp),dx=1) 
                tmp = [0]
        if tmp != [0]:

            tmp.append(0)
            score += simps(np.array(tmp),dx=1)
        scores.append(score)
    scores = np.array(scores)
    return scores

def combined():
    y = wPrime()#POgiven()
    d, groups = descriptors()
    # print(len(~np.isnan(d).any(axis=1)))
    nonEmptyIndexs = ~np.isnan(d).any(axis=1)
    y= y[nonEmptyIndexs]
    # y = binValues(y,5)
    # print(y)
    # print(~np.isnan(d).any(axis=1))
    d= d[nonEmptyIndexs]
    groups = groups[nonEmptyIndexs]
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)
    for i in range(1,120): #(2,100):
        c, peaksC  =  Cadenece(i)
        p, peaksPO = PO(i)
        v, peaksVO = VO2(i)
        # print(len (c))
        c = c[nonEmptyIndexs]
        p = p[nonEmptyIndexs]
        v = v[nonEmptyIndexs]
        peaksPO = peaksPO[nonEmptyIndexs]
        peaksVO = peaksVO[nonEmptyIndexs]
        peaksC = peaksC[nonEmptyIndexs]
        # X = d
        # print(peaksC)
        # X = np.concatenate((peaksPO, peaksC, peaksVO), axis=1) # c,p,v
        X = np.concatenate((d,peaksPO, peaksC, peaksVO), axis=1) # c,p,v
        from sklearn.decomposition import PCA
        pca = PCA(n_components=4)
        X = pca.fit_transform(X)

        idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
        train_idx = idx[0]
        test_idx = idx[1]
        X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx] #, random_state=0)
        # print(y_train)
        # print(y_test)
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)#, random_state=0)
        reg = Ridge().fit(X_train, y_train) #Ridge
        plt.scatter(i+5,reg.score(X_test, y_test), c='black')
    plt.xlabel('Time used')
    plt.ylabel('Score')
    plt.title('Ridge regression - Combined Features')
    plt.show()

# combined()
def oversampled():
    import smogn
    y = POgiven()
    d, groups = descriptors()
    nonEmptyIndexs = ~np.isnan(d).any(axis=1)
    y= y[nonEmptyIndexs] 
    d= d[nonEmptyIndexs]
    groups= groups[nonEmptyIndexs]
    X = d#np.concatenate((d), axis=1)
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)

    idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
    train_idx = idx[0]
    test_idx = idx[1]
    X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx] #, random_state=0)

    com = np.append(X_train,[[Y] for Y in y_train], axis=1)
    df = pd.DataFrame(com,columns=['X1','X2','X3','X4','X5','X6','X7','X8','CP'])

    while True:
        try:
            X_smogn = smogn.smoter(
                
                data = df, 
                y = 'CP' 
            )
            break

        except:
            print("Trying to resample")
            gss = GroupShuffleSplit(n_splits=1, train_size=.6)

            idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
            train_idx = idx[0]
            test_idx = idx[1]
            X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx] #, random_state=0)

            com = np.append(X_train,[[Y] for Y in y_train], axis=1)
            df = pd.DataFrame(com,columns=['X1','X2','X3','X4','X5','X6','X7','X8','CP'])

    y_train = np.array(X_smogn['CP'])
    X_train = np.array(X_smogn[['X1','X2','X3','X4','X5','X6','X7','X8']])
    reg = Ridge().fit(X_train, y_train) #Ridge
    print(reg.score(X_test,y_test))

def baselineModel():
    ... # create and save # repeat for multiple runs # choose best
    # if exists just load model
    # just decriptors
# combined()

# Tasks:

#formular for CP - model to workout formular unsupervisored 
# drop in error graph
# pca on PO
# implement other score/error metrics
# make one model 'baseline' - just descriptors (as this uses none of 3mt data)
# ridge regression and other types look at
# implement neural networks
# start report - rewrite introduction, related work
# start presentation
# look at research practice and professionalism



# Notes:
# given peaks are from whole test so cant be used - we also dont have data they are generated from
# standard score - 'name' - is the amount of variance that can be explained by the model
# w' is the area under the curve above this critical power