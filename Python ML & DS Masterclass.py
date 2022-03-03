# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 22:43:42 2022

@author: akoh4
"""

#=======NUMPY========#
#Numpy; library for creating N-dimensional arrays. Contains built-in linear algebra, statistical distributions, trigonometric and random number capabilities
#NOTE:Numpy is very useful for quickly applying functions to our data sets
import numpy as np
mylist = [1,2,3]
myarr = np.array(mylist)

my_matrix = [[1,2,3],[4,5,6],[7,8,9]]
my_matrix_arr = np.array(my_matrix)
my_matrix_arr

np.arange(0,10,2)   #arange() sets the range specified in the brackets eg; arange(0,10,2) prints the range of numbers from 0 to 10 printing every 2nd digit
np.zeros((5,5)) #assigns zero to each array item. 1st elements in array is Rows and 2nd element is columns
np.ones((5,5)) #assigns one to each array item. 1st elements in array is Rows and 2nd element is columns
np.linspace(0,10,11) #(linearly space)evenly spaces numbers between numbers eg; linspace(0,10,20) prints the range 0 to 10 with 20 digits spaced evenly inbetween
np.eye(5) #identity matrix based on square matrix

np.random.rand(5,4) #random number between 0 & 1  based on a uniform distirbution(same likelyhood of being selected) based on 5x4 array
np.random.randn(10,10) #used to creat sample from a standard normal distribution (mean = 0 & variance = 1) NOTE: values closer to zero are more likely to appear, based on a 10x10 array
np.random.randint(0,101,10) #same as .rand but only returns integers
np.random.seed(42)  #lets you select a particular set of random numbers(always select same sample of ranodm numbers). Choosing arbitrary seed number to get a set of random numbers
np.random.rand(4)
arr = np.arange(0,25)
arr 
arr.reshape(5,5)

ranarr = np.random.randint(0,101,10)
ranarr.max() #returns max value in array
ranarr.min() #returns min value in array
ranarr.argmax() #returns index of max value
ranarr.argmin() #returns index of min value
ranarr.dtype # returns data type
arr.shape #.shape returns the dimensions of an array
arr.reshape #chnages the dimensions of an array

#Challenge
myarray = np.linspace(0,10,101)
myarray

#Numpy Indexing and Selection
arr = np.arange(0,11)
arr[8] #selext value in array based on index
arr[1:5] #select between index
arr[:5] #selects from 1st to 5th index
arr[5:] #selects from 5th to last index

#Numpy Broadcasting
arr[0:5] = 100 #chnage values in array by specifying index location

#Numpy Slicing
slice_arr = arr[0:5] # subsetting array
slice_arr[:] #selects everything iin array
arr_cop=arr.copy() #copies vaue of an array

#Numpy 2D Array
arr_2d = np.array([[5,10,15],[20,25,30],[35,40,45]])
arr_2d[1]
arr_2d[1][2] #1st [] is for Row subset 2nd [] is for Column subset
arr_2d[:2,1:] #1st [:2,] is for Row subset 2nd [,1:] is for Column subset

#Conditional Selection
arr = np.arange(0,11)
arr > 4
arr

#Challenge
dice_rolls = np.array([3, 1, 5, 2, 5, 1, 1, 5, 1, 4, 2, 1, 4, 5, 3, 4, 5, 2, 4, 2, 6, 6, 3, 6, 2, 3, 5, 6, 5])
total_rolls_over_two = len(dice_rolls[dice_rolls >2])
total_rolls_over_two

#Numpy Operations
arr = np.arange(0,10)
arr + 5
arr + arr
arr * arr
arr/arr
np.sqrt(arr)
np.sin(arr)
np.log(arr)
np.sum(arr)
np.mean(arr)
np.max(arr)
np.var(arr)
np.std(arr)

arr2d=np.arange(0,25).reshape(5,5)
arr2d.sum()
arr2d.sum(axis=0) # axis=0 means perform sum operation across all rows & axis=1 performs sum operations across Columns

#Challenge
account_transactions = np.array([100,-200,300,-400,100,100,-230,450,500,2000])
np.sum(account_transactions)

arr=np.arange(0,11)
arr[:]= 1
arr

np.arange(10,50)
np.arange(10,52,2)
np.arange(0,9).reshape(3,3)

np.random.rand(1)
np.random.randn(25)
np.linspace(0,1,100)
np.linspace(0,1,20)
mat = np.arange(1,26).reshape(5,5)
mat
mat[2:,1:]
mat[3,4]
mat[:3,1:2]
mat[4]
mat[3:]
np.sum(mat)
np.std(mat)
mat.sum(axis=0)


#=======PANDAS========#
#Data analysis tool built off of NumPy
#Series is a data structure that holds an array of information along with Named Index- which is the difference from Numpy array. ALso a one dimensional NDarray with axis labels
import pandas as pd
help(pd.Series)

myindex = ['USA', 'Canada','Mexico']
mydata = [1776,1867,1821]
myser = pd.Series(data=mydata, index=myindex)
ages = {'Sam':5,'Frank':10,'Spike':7}
pd.Series(ages) #transforms dictionary into a series

#Challenge
expenses = pd.Series({'Andrew':200,'Bob':150,'Claire':450})
bob_expense = expenses['Bob']
bob_expense

q1 = {'Japan':80,'China':450,'India':200,'USA':250}
q2 = {'Brazil':100,'China':500,'India':210,'USA':260}
sales_q1 = pd.Series(q1)
sales_q2 = pd.Series(q2)
sales_q1['Japan'] #selects the corresponding column value based on index selected
sales_q2['Brazil']
sales_q1.keys() #.keys() returns the index objects of your series
sales_q1 * 3 #you can perform numerical operations on a series
sales_q1 + sales_q2 #inthis case pandas returns values for items present in both series and returns NULL for items not present in both
sales_q1.add(sales_q2, fill_value=0 )

#DataFrames is a table of columns and rows in pandas that can easily be restructured and filtered. (Group of Pandas Series objects that share the same index)
np.random.seed(101)
mydata = np.random.randint(0,101,(4,3))
mydata
myindex = ['CA','NY','AZ','TX']
mycolumns = ['Jan', 'Feb','Mar']
df = pd.DataFrame(data = mydata,index = myindex, columns = mycolumns) #creates dataframes from multiple series with ability to specify indexes
pwd #prints out your working directory
ls #prints out all files in your current working directory
import os #working directory library
os.chdir('C:\\Users\\akoh4\\Documents\\Pyhton\\UNZIP_FOR_NOTEBOOKS_FINAL\\03-Pandas') #os.chdir sets the working directory to specified path
df = pd.read_csv('tips.csv') #.read_csv reads in excel csv files in to python. this uses '\\' eg 'C:\\Users\\akoh4'
df
df.columns      #returns columns headers
df.index        #returns values for index
df.head(10)     #returns df with top rows based on specified number (SQL equaivalent is LIMIT)
df.tail(10)     #returns df with last riws based on specified number
df.info()       #returns information about df
df.shape        #returns the number of rows and columns
df.describe()   #returns the statistical profile(mean, count, std,min/max,percentile) of the dataframe (SQL equivalent of aggregate functions)
df.describe().transpose() #trnsposes index and columns (moves index into columns and columns to index)

df['total_bill'] #grabs single column
mycols =['total_bill', 'tip']
df[mycols]
df[['total_bill', 'tip']]

df['tip'] / df['total_bill'] *100 #performs numerical operations on dataframe columns
df['tip_percentage'] = 100 * df['tip'] / df['total_bill'] #creates and adds new column to df based on numerical operations performed
df['price_per_person'] = df['total_bill'] / df['size']
df['price_per_person'] =np.round(df['total_bill'] / df['size'], 2) #np.round() rounds values to sepcified digits
df.drop('tip_percentage', axis =1, inplace=True) #.drop() deletes columns specified ONLY by setting Axis = 1. Setting 'inplace = True' permanently removes it from the df

df.set_index('Payment ID', inplace=True) #sets specifiefd column as index
df.reset_index()  #resets index columns

df.iloc[0:4] #returns row based on position of row index specified
df.loc[['Sun2959','Sun5260']] #returnds rows based on name of row index specified
df.drop('Sun2959', axis = 0) #.drop deletes specified row ONLY by setting Axis = 0. Setting 'inplace = True' permanently removes it from the df
one_row = df.iloc[0]
df=df.append(one_row)

#Conditional Filtering; lets you select rows based on a condition on a column
#Note: Columns are features(attributes/properties of rows). Rows are instances of data
df = pd.read_csv('tips.csv')
bool_series=df['total_bill'] >40
df[bool_series]
df[df['total_bill'] >40] #filters based off logic specified
df[df['sex'] == 'Male']
df['total_bill'] > 30
df['sex'] == 'Male'
df[(df['total_bill'] > 30) & (df['sex'] == 'Male')]  #using '&' to filter multiple columns BOTH conditions need to be true
df[(df['total_bill'] > 30) | (df['sex'] == 'Male')]  #using '|' to filter multiple columns EITHER conditions need to be true
df[(df['day']=='Sun') | (df['day']=='Sat') | (df['day']=='Fri') ]

options = ['Sat','Sun']
df['day'].isin(options)  #.isin() checks if list specified in df and returns boolean value for each row. NOTE:this is an alternative to using the '|' for filtering
df[df['day'].isin(['Sat','Sun'])]

#Useful Methods
#Apply Function
str(12345675634)[-4:]  #converts elements to string and slices
def last_four(num):      #function to convert any number to string and return last four digits
    return str(num)[-4:]
last_four(12345675634)
df['last_four'] = df['CC Number'].apply(last_four)  #.apply() applies a function to item(s) in df and adds a new column to a df

df['total_bill'].mean()
def yelp(price):        #function returns '$' based on conditions
    if price < 10:
        return '$'
    elif price >= 10 and price < 30:
        return '$$'
    else:
        return '$$$'
df['yelp']=df['total_bill'].apply(yelp)
df

#Lambda Expression
def simple(num):
    return num*2
lambda num: num*2 #function converted to lambda
df['total_bill'].apply(lambda num: num*2)

def quality(total_bill, tip):
    if tip/total_bill > 0.25:
        return 'Generous'
    else:
        return 'Other'
df['Quality']= df[['total_bill','tip']].apply(lambda  df: quality(df['total_bill'],df['tip']),axis = 1)

#np.vectorize: transforms functions which are not numpy aware (takes floats as input and return them as outputs) into functions that can operate on and return numpy arrays
df['Quality'] = np.vectorize(quality)(df['total_bill'],df['tip'])  #np.vectorize helps apply functions involving multiple columns

#Statistical Information and Sorting DataFrames
df = pd.read_csv('tips.csv')
df.sort_values(['tip','size'], inplace=True) #sorts entire df based on specified column 
df['total_bill'].max() #returns max value of a column
df['total_bill'].idxmax() #returns index location of max value
df.corr()  #.corr() returns the pearson correlation coefficient between each column variable 

df['sex'].value_counts()  #.value_counts() counts the number of EACH distinct variables and returns the count
df['day'].unique()  #.unique() returns distintcs items of a column in a df (SQL equivalent of DIST      INCT)
df['day'].nunique() #.nunique returns the number of unique items in a df (equivalent of using 'len' function)

df['sex'].replace(['Female','Male'],['F','M']) #.replace)() replaces items in a column based on specified values
mymap = {'Female':'F','Male':'M'}  #.map() assigns items from a dictionary to a df
df['sex'].map(mymap)

simple_df = pd.DataFrame([1,2,2,2],['a','b','c','d'])
simple_df.duplicated()   #checks if df has duplicate rows
simple_df.drop_duplicates()  #deletes duplicate rows

df[df['total_bill'].between(10,20,inclusive=True)] #.between() returns boolean value based on range specified (SQL equivalent is BETWEEN)
df.nlargest(10, 'tip')  #returns the number of rows ordered by specified column (SQL equivalent of ORDER BY DESC and LIMIT BY).
df.nsmallest(10, 'tip')  #returns the number of rows ordered by specified column (SQL equivalent of ORDER BY ASC and LIMIT BY).
df.sample(frac=0.1)     #returns a sample of n number of random rows specified. 'frac' helps sample based on a fraction of the entire df

#Missing Data
    #Keep Data
    #Drop Data
    #Fill Data
np.nan
df = pd.read_csv('movie_scores.csv')
df.isnull()   #returns a boolean for all items in df if there is a null value. opposite method is '.notnull()'
df[df['pre_movie_score'].notnull()]  #conditional filtering
df[(df['pre_movie_score'].isnull()) & (df['first_name'].notnull())]

#Drop Data
df.dropna()  #drops all rows with missing values
df.dropna(thresh =1)  #drop any rows with null UNLESS they contain atleast 1 null value
