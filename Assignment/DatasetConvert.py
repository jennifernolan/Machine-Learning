'''
    Jennifer Nolan
    C16517636
    Program to generate the continuous and categorical feature tables in a Data Quality Report
'''

#import the following libraries
import pandas as pd
import numpy as np

def main():

    #read the feature names from the specified path
    featureNames = pd.read_csv("./data/feature_names.txt", header = None)
    
    #read the .csv file from the specified path
    df = pd.read_csv("./data/dataset.csv")
    
    #place headers on the columns of the dataset from the .csv
    df.columns = ['id', 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'target']
    
    #place the continuous headers into a list
    cont = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    
    #create a list for each column that will be in the continuous table
    count, percentMiss, card, mini, firstQrt, avg, med, thirdQrt, maxi, stdDev = [], [], [], [], [], [], [], [], [], []
    
    #loop through the continuous header list and add the values of the functions to the corresponding list
    for i in range(0, len(cont)):
        count.append(df[cont[i]].count()) #count
        percentMiss.append((((df[cont[i]].values == 0).sum()) / count[0] * 100).round(3)) #percentage of missing values
        card.append(df[cont[i]].nunique()) #cardinality
        mini.append(df[cont[i]].min()) #the minimum value
        firstQrt.append(df[cont[i]].quantile(0.25)) #first quartile
        avg.append(df[cont[i]].mean()) #the mean
        med.append(df[cont[i]].median()) #middle value when ordered
        thirdQrt.append(df[cont[i]].quantile(0.75)) #third quartile
        maxi.append(df[cont[i]].max()) #the maximum value
        stdDev.append(df[cont[i]].std()) #standard deviation
        
    #place all the values gathered in the loop into a dataframe
    continuous = pd.DataFrame({'FEATURENAME': cont, 'Count': count, '% Miss': percentMiss, 'Card.': card, 'Min': mini, '1st Qrt': firstQrt, 'Mean': avg, 'Median': med, '3rd Qrt': thirdQrt, 'Max': maxi, 'Std. Dev': stdDev})
    
    #write the continuous dataframe to a new file with the specified name
    continuous.to_csv("C16517636CONT.csv", index = False)
    
    #place the categorical headers into a list
    cat = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'target']
    
    #create a list for each column that will be in the categorical table
    counter, perMiss, cardi, mod, modeFreq, modePer, secMode, secModeFreq, secModePer = [], [], [], [], [], [], [] , [], []
    
    #loop through the categorical header list and add the values of the functions to the corresponding list
    for i in range(0, len(cat)):
        counter.append(df[cat[i]].count()) #count
        perMiss.append((((df[cat[i]].values == ' ?').sum()) / count[0] * 100).round(3)) #percentage of missing values specified with a ?
        cardi.append(df[cat[i]].nunique()) #cardinality
        mod.append(df[cat[i]].mode()[0]) #mode (most common)
        modeFreq.append(df[cat[i]].value_counts()[0]) #the frequency of the mode
        modePer.append(df[cat[i]].value_counts(normalize = True).round(3)[0] * 100) #the percentage of the mode
        secMode.append(df[cat[i]].value_counts().index[1]) #second mode (second most frequent value
        secModeFreq.append(df[cat[i]].value_counts()[1]) #second mode frequency
        secModePer.append(df[cat[i]].value_counts(normalize = True).round(3)[1] * 100) #second mode percentage
    
    #place all the values gathered in the loop into a dataframe
    categorical = pd.DataFrame({'FEATURENAME': cat, 'Count': counter, '% Miss': perMiss, 'Card.': cardi, 'Mode': mod, 'ModeFreq': modeFreq, 'Mode %': modePer, '2nd Mode': secMode, '2nd Mode Freq': secModeFreq, '2nd Mode %': secModePer})
    
    #write the categorical dataframe to a new file with the specified name
    categorical.to_csv("C16517636CAT.csv", index = False)


main()
