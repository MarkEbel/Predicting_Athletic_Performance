
# function - read csv into dataframe
# make function to compute derivatives at different time slots
# function - save to csv
from cProfile import label
from black import out
from click import option
import pandas as pd
import numpy as np
from sqlalchemy import over, true
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
    X,groups = descriptors()
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
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)
    for j in range(iterations):
        for i in range(1,100): #(2,100):
            X, peaks  =  PO(i) # VO2(i)np.concatenate(POplotted(i), CadenecePlotted(i))7
            d , groups = descriptors()
            idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
            train_idx = idx[0]
            test_idx = idx[1]
            X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx]
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
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)
    for i in range(24,100): #(2,100):
        X, peaks  =  VO2(i) # VO2(i)np.concatenate(POplotted(i), CadenecePlotted(i))7
        d , groups = descriptors()
        idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
        train_idx = idx[0]
        test_idx = idx[1]
        X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx]
        reg = LinearRegression().fit(X_train, y_train)
        plt.scatter(i+2,reg.score(X_test, y_test), c='black')
    plt.xlabel('Time used')
    plt.ylabel('Score')
    plt.title('Linear regression - VO2')
    plt.show()


def Cgraph():
    y = POgiven()
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)
    for i in range(1,100): #(2,100):
        X, peaks  =  Cadenece(i) # VO2(i)np.concatenate(POplotted(i), CadenecePlotted(i))7
        d , groups = descriptors()
        idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
        train_idx = idx[0]
        test_idx = idx[1]
        X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx]
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
                    tmp.append(float(0.0))
                    continue
            if(sum(np.array(tmp)) >0):
                der = compute_derivative(tmp[0:amount],1)
                der = compute_derivative(der,2)
            outStuff.append(der)

            peaks.append([np.max(np.array(tmp))])
    return np.array(outStuff), np.array(peaks)


def saveModel(model, name):
    from joblib import dump
    dump(model, name+'.joblib')
def loadModel(loc):
    from joblib import load
    try:
        return load(loc+'.joblib')
    except:
        return "No Model"
def binValues(vals, binSize):
    b = np.digitize(vals, range(int(np.floor(min(vals))),int(np.ceil(max(vals))),binSize))
    b = int(np.floor(min(vals))) + binSize/2 + (b-1)*binSize
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

def oversampledGraph():
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
    print(RMSE(reg,X_test,y_test))
    print(MAE(reg,X_test,y_test))
    # return X_train,y_train,X_test,y_test
def oversampled(X,y):
    import smogn
    d, groups = descriptors()
    nonEmptyIndexs = ~np.isnan(d).any(axis=1)
    groups= groups[nonEmptyIndexs]
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)

    idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
    train_idx = idx[0]
    test_idx = idx[1]
    X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx] #, random_state=0)

    r = np.arange(len(X_train[0]))
    a = np.append(r,'CP')
    com = np.append(X_train,[[Y] for Y in y_train], axis=1)
    df = pd.DataFrame(com,columns=a)
    # above line needs to be changed to work with variety of column nums
    # print(df)
    # X_smogn = smogn.smoter(
    #         data = df, 
    #         y = 'CP' 
    #     )
    while True:
        try:
            X_smogn = smogn.smoter(
                rel_coef=0.5,
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
            df = pd.DataFrame(com,columns=a)
            
    r = [str(R) for R in r]
    y_train = np.array(X_smogn['CP'])
    X_train = np.array(X_smogn[r])
    # print((y_train.shape))
    # print((X_train.shape))
    return X_train,y_train,X_test,y_test

def baselineModel(w,ignore=False, iterations=100):
    bl = loadModel("baselineCP")
    y = POgiven()
    if w:
        bl = loadModel("baselineWprime")
        y = wPrime()
    d, groups = descriptors()
    nonEmptyIndexs = ~np.isnan(d).any(axis=1)
    y= y[nonEmptyIndexs] 
    d= d[nonEmptyIndexs]
    groups= groups[nonEmptyIndexs]
    if bl == "No Model" or ignore:

        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, train_size=.6)
        idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(d, y, groups)][0]
        train_idx = idx[0]
        test_idx = idx[1]
        X_train, X_test, y_train, y_test = d[train_idx],d[test_idx],y[train_idx],y[test_idx] #, random_state=0)
        bestM = Ridge().fit(X_train, y_train)
        bScore = bestM.score(X_test,y_test)
        for i in range(iterations):
            gss = GroupShuffleSplit(n_splits=1, train_size=.6)
            idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(d, y, groups)][0]
            train_idx = idx[0]
            test_idx = idx[1]
            X_train, X_test, y_train, y_test = d[train_idx],d[test_idx],y[train_idx],y[test_idx] #, random_state=0)
            
            m = Ridge().fit(X_train, y_train)
            if(m.score(X_test,y_test)>bScore):
                bestM = m
                bScore = bestM.score(X_test,y_test)
        if w:
            saveModel(bestM, "baselineWprime")
            print(bScore)
            return bestM
        else:
            saveModel(bestM, "baselineCP")

            print(bScore)
            print(RMSE(bestM,X_test,y_test))
            print(MAE(bestM,X_test,y_test))
            return bestM
    else:
        return bl

def MSE(model,X,y):
    predY = model.predict(X)
    return sum((y-predY)**2)/len(y)

def RMSE(model,X,y):
    return (MSE(model,X,y))**(1/2)

def MAE(model,X,y):
    predY = model.predict(X)
    return sum(np.abs(y-predY))/(len(y))

def MAEP(model,X,y):
    predY = model.predict(X)
    return sum(np.abs((y-predY)/y)*100)/(len(y))

def combined(w, iterations = 2,binned = False, pCA = False, optionX = 0,optionScore = 1,saveOrShow = True, os = False) :
    y = POgiven()
    if w:
        y = wPrime()
    d, groups = descriptors()
    nonEmptyIndexs = ~np.isnan(d).any(axis=1)
    y= y[nonEmptyIndexs]
    d= d[nonEmptyIndexs]
    groups = groups[nonEmptyIndexs]
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, train_size=.6)
    for j in range(iterations):
        for i in range(1,150): #(2,100):
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
            if optionX == 0:
                X = np.concatenate((d,peaksPO, peaksC, peaksVO,p,c), axis=1) # c,p,v # best for W prime
            elif optionX == 1:
                X = np.concatenate((d,peaksPO, peaksC, peaksVO), axis=1) # c,p,v
            elif optionX == 2:
                X = np.concatenate((p,d), axis=1) # c,p,v
                
            if pCA:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=len(X[0]))
                if(len(X[0])) >= 207:
                    break
                X = pca.fit_transform(X)
                

            idx = [[train_idx, test_idx] for train_idx, test_idx in gss.split(X, y, groups)][0]
            train_idx = idx[0]
            test_idx = idx[1]
            X_train, X_test, y_train, y_test = X[train_idx],X[test_idx],y[train_idx],y[test_idx] 

            if binned:
                y_train = binValues(y_train,5)
            if os:
                import sys
                import os

                X_train,y_train,X_test,y_test = oversampled(X,y)


            reg = Ridge().fit(X_train, y_train) 
            if optionScore == 0:
                plt.scatter(i+5,RMSE(reg,X_test, y_test), c='black')
            elif optionScore == 1:
                plt.scatter(i+5,reg.score(X_test, y_test), c='black')
            elif optionScore == 2:
                plt.scatter(i+5,MAE(reg,X_test, y_test), c='black')
            elif optionScore == 3:
                plt.scatter(i+5,MAEP(reg,X_test, y_test), c='black')
    plt.xlabel('Time used')
    plt.ylabel('Score')
    t = 'Ridge regression - Combined Features - '
    t = t+'i' + str(iterations)
    t = t+'o' + str(optionX)
    t = t+'s' + str(optionScore)
    
    if w:
        t= t+'W'
    else:
        t= t+'CP'
    if binned:
        t = t+'b'
    if pCA:
        t = t+'p'
    if os:
        t = t+'OS'

    plt.title(t) # i2, b, p, o1,s1,OS = i2bp01s1os - add to title/ generate name
    if(saveOrShow):
        plt.show()
    else:    
        
        plt.savefig(t + 'combined.png')

# combined(True,iterations=3,optionX=0, optionScore=3)
def descriptorClusters():
    des, groups = descriptors()
    p = POgiven()
    for d in range(des.shape[1]):
        plt.scatter(des[:,d],p)
        plt.savefig(str(d )+ 'descriptor.png')#
        plt.figure()

descriptorClusters()
# combined(True, iterations = 30,binned = False, pCA = False, optionX = 0,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = True, pCA = False, optionX = 0,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = False, pCA = True, optionX = 0,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = False, pCA = False, optionX = 0,optionScore = 3,saveOrShow = False, os = True)
# combined(True, iterations = 30,binned = False, pCA = False, optionX = 1,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = True, pCA = False, optionX = 1,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = False, pCA = True, optionX = 1,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = False, pCA = False, optionX = 1,optionScore = 3,saveOrShow = False, os = True)
# combined(True, iterations = 30,binned = False, pCA = False, optionX = 2,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = True, pCA = False, optionX = 2,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = False, pCA = True, optionX = 2,optionScore = 3,saveOrShow = False, os = False)
# combined(True, iterations = 30,binned = False, pCA = False, optionX = 2,optionScore = 3,saveOrShow = False, os = True)

# combined(False, iterations = 30,binned = False, pCA = False, optionX = 0,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = True, pCA = False, optionX = 0,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = False, pCA = True, optionX = 0,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = False, pCA = False, optionX = 0,optionScore = 3,saveOrShow = False, os = True)
# combined(False, iterations = 30,binned = False, pCA = False, optionX = 1,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = True, pCA = False, optionX = 1,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = False, pCA = True, optionX = 1,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = False, pCA = False, optionX = 1,optionScore = 3,saveOrShow = False, os = True)
# combined(False, iterations = 30,binned = False, pCA = False, optionX = 2,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = True, pCA = False, optionX = 2,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = False, pCA = True, optionX = 2,optionScore = 3,saveOrShow = False, os = False)
# combined(False, iterations = 30,binned = False, pCA = False, optionX = 2,optionScore = 3,saveOrShow = False, os = True)

# Tasks:
# explore what descriptors are most useful
# equalise classes for sex for test and train split
# clustering of desciptors see any groups
# use less for loops when calculating values and speed up code
# dont run whole for 30 iterations just 5-35 seconds

# find out current error 3%?
# redo all graph images 
# need to use combined to suggest a model aka use 20 secs of data and then generate and save best model
# implement neural networks
#formular for CP - model to workout formular unsupervisored 
# start report - rewrite introduction, related work
# start presentation
# start project logbook - research practice and professionalism



# Notes:
# given peaks are from whole test so cant be used - we also dont have data they are generated from
# standard score - 'name' - is the amount of variance that can be explained by the model
# w' is the area under the curve above this critical power
# W Prime has lot more needing predicting baseline is pretty good for CP



# baselineModel(True,True,100000)
# Descriptorgraph(100)
# POgraph(2)
# VO2graph()
# Cgraph()
# POplotted()
# VO2_plotted()

