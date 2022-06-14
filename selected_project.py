# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 14:10:18 2018

@author: hp
"""
import pandas as pd              # Data analysis library
import seaborn as sns            #DV library
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean 
win=pd.read_csv("dataset.csv",delimiter=";")
print(win.head())
print(win.describe())
#1
#rug plots
sns.distplot(win['mpg'], rug=True, hist=False)
plt.savefig('rugmpg.png')
plt.show()
sns.distplot(win['cylinders'], rug=True, hist=False)

plt.savefig('rugcy.png')
plt.show()
sns.distplot(win['acceleration'], rug=True, hist=False)

plt.savefig('rugacc.png')
plt.show()
sns.distplot(win['displacement'], rug=True, hist=False)

plt.savefig('rugdis.png')
plt.show()
sns.distplot(win['origin'], rug=True, hist=False)
plt.savefig('rugori.png')
plt.show()
#scatter plot
pd.tools.plotting.scatter_matrix(win,alpha=0.4,range_padding=0.7,figsize= (10,10))
plt.savefig('scattermatrix.png')
plt.show()
#histogram

plt.hist(win["mpg"],)
plt.title("mpg Histogram")
plt.xlabel("mpg")
plt.ylabel("Frequency")
plt.savefig('histmpg.png')
plt.show()


plt.hist(win["acceleration"],bins=15)
plt.title("acceleration Histogram")
plt.xlabel("acceleration")
plt.ylabel("Frequency")
plt.savefig('hisacc.png')
plt.show()

plt.hist(win["cylinders"],bins=15)
plt.title("cylinders Histogram")
plt.xlabel("cylinders")
plt.ylabel("Frequency")
plt.savefig('hiscy.png')
plt.show()

plt.hist(win["displacement"],bins=15)
plt.title("displacement Histogram")
plt.xlabel("displacement")
plt.ylabel("Frequency")
plt.savefig('hisdis.png')
plt.show()

plt.hist(win["origin"],bins=15)
plt.title("origin Histogram")
plt.xlabel("origin")
plt.ylabel("Frequency")
plt.savefig('hisor.png')
plt.show()
#box plot


sns.boxplot(data=win['mpg'], orient="h", width=0.2)
plt.xlabel("mpg")
plt.savefig('boxmpg.png')
plt.show()

sns.boxplot(data=win['displacement'], orient="h", width=0.2)
plt.xlabel("displacement")
plt.savefig('boxdispl.png')
plt.show()

sns.boxplot(data=win['cylinders'], orient="h",width=0.2)
plt.xlabel("cylinders")
plt.savefig('boxcy.png')
plt.show()

sns.boxplot(data=win['origin'], orient="h", width=0.2)
plt.xlabel("origin")
plt.savefig('boxori.png')
plt.show()



#2
print("the mean of dataset\n",win.mean(),"the var is \n",win.var()) #2


#3
mu=win['mpg'].mean()
sigma=win['mpg'].var()
list1=[]
for i in range(1,50):
    number_observations=10
    y=[1]*number_observations;
    #Generate observations randomly from N(mu,sigma)
    observations = np.random.normal(mu,sigma,number_observations)
    print("mean of sample",i,observations.mean())
    print("std of sample",i,observations.var())
    list1.append(observations.mean())
print("\nthe mean of samples means",mean(list1))

#4 cov to each pair of dataset
sns.heatmap(win.cov(),annot=True,cmap="YlGnBu",fmt=".2f",linewidth=3)
plt.savefig('heatmap.png')
plt.show()
#pd.tools.plotting.scatter_matrix(win,alpha=0.2,range_padding=0.7,figsize= (10,10))





##5
#the coveriance matrix is to show the Association relation between each two pairs 
#all the inverse relation have a Negative sign in covariance matrix 
#all the a direct relationship have a positive sign in covariance matrix
#between mpg and (cylinders,displacement)Inverse relation so the cov have a Negative sign
#covariance  numbers do not explicitly indicate the strength of the relation (model ,year ),(mpg,year)
#to get indicate relation number get the Correlation 
#covariance matrix numbers != 0  because the dateset is depende and liner 


#
##6
#mean of samples means :
#    it is average of samples 
#    The sample mean  from a group of observations is an estimate of the population mean 
#    given a sample of size n, consider n independent random variables X1, X2, ..., Xn, each corresponding to one randomly selected observation. Each of these variables has the distribution of the population,
#    with mean mu   and varianc sigma (like point 3)
#    _
#    x=1/n(X1, X2, ..., Xn)
#    if number of observations is increased the sample mean is =~ mean of dateset 
#mean of dataset :
#    is integration of pdf   
    

'''
8

with ploting samples with there numbers with larg number of observition the mean of  
all samples will get closed to the mean of dataset
the varince of all samples 




'''


















##7
#
#sns.distplot(win.mean(), kde=True, rug=False, hist=False, rug_kws={"color": "g"})
#for i in range(1,50):
#    number_observations=10
#    y=[1]*number_observations;
#    #Generate observations randomly from N(mu,sigma)
#    observations = np.random.normal(mu,sigma,number_observations)
#    sns.distplot(observations.mean(), kde=True, rug=False, hist=False, rug_kws={"color": "g"})

#
#df_mean = pd.DataFrame(samples_mean)
#df_mean['drinks'].plot.kde(label='drinks sample Mean')
#samples[0]['drinks'].plot.kde(label='drinks Sample')
#plt.legend(loc='upper right')
#plt.show()
#plt.clf()
#
#
