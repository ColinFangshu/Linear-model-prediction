# CEE 5930 hw3
# Fang Shu (fs444)
# Oct 17, 2019
# Linear model prediction 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

import statsmodels.api as sm

print("-----The first problem-----")
# import the data
df_demand_temp = pd.read_excel('../CEE 5930 Assignment 3 Data -- Fall 2019.xlsx', sheet_name='Demand Data')

X = df_demand_temp['Daily High Temp (F)']
Y = df_demand_temp['Peak Demand (MwH)']
X = sm.add_constant(X)
model1 = sm.OLS(Y,X)
results1 = model1.fit()

print("The linear regression model results of the first dataset")
print(results1.summary())

print("The degree of freedom of the residuals: {}".format(results1.df_resid))
print("The degree of freedom of the regression: {}".format(results1.df_model))
print("The sum of squares (SS) of regression: {}".format(results1.ess))
print("The Regression SS / Regression degrees of freedom: {}".format(results1.mse_model))
print("The mean squared error (Residual SS / Residual degrees of freedom): {}".format(results1.mse_resid))
print("The sum of squares (SS) of residuals: {}".format(results1.ssr))
print("The sum of squares (SS) of total: {}".format(results1.centered_tss))
print("The multiple R: {}".format(np.sqrt(results1.rsquared)))
print("The standard error: {}".format(np.std(results1.resid)))

# calculate a 95% confidence interval for a prediction of peak power demand on a day when the forecast high temperature is 88 F
slope = 1.5295
intercept = -74.9047
y_hat = intercept + slope*df_demand_temp['Daily High Temp (F)']
y_hat_x_equal_88 = intercept + slope*88
t_value = 2.056
n = 28
k = 1
x_mean = np.mean(df_demand_temp['Daily High Temp (F)'])
Se = np.sqrt(np.sum(np.square(df_demand_temp['Peak Demand (MwH)']-y_hat))/(n-k-1))
Sxx = np.sum(np.square(df_demand_temp['Daily High Temp (F)']-x_mean))

interval_with_95_confidence_when_x_equals_88 = [y_hat_x_equal_88 - t_value*Se*np.sqrt(1+1/n+(np.square(88-x_mean)/Sxx)),
                                                y_hat_x_equal_88 + t_value*Se*np.sqrt(1+1/n+(np.square(88-x_mean)/Sxx))]

print("The 95% confidence interval for a prediction of peak power demand on a day when the forecast high temperature is 88 F is: {}".format(interval_with_95_confidence_when_x_equals_88))

# Compute the PRESS statistic and the coefficient of prediction (P2) for this regression
hii = 1/n + np.square(df_demand_temp['Daily High Temp (F)']-x_mean)/np.sum(np.square(df_demand_temp['Daily High Temp (F)']-x_mean))
TSS = results1.centered_tss

# calculate from the R 
PRESS = 382.0537

P_squared = 1- (PRESS/TSS)*np.square((n-1)/n)
print("The P squared value of the simple linear regression model is :{}".format(P_squared))


print("-----The second problem-----")

# import the data
df_maint_cost = pd.read_excel('../CEE 5930 Assignment 3 Data -- Fall 2019.xlsx', sheet_name='Maint Cost Data')

X = df_maint_cost["Customers (000's)"]
Y = df_maint_cost['Line Maintenance ($000)']
X = sm.add_constant(X)
model2 = sm.OLS(Y,X)
results2 = model2.fit()
results2.params

y_hat = df_maint_cost["Customers (000's)"]*results2.params[1] + results2.params.const
residual = df_maint_cost["Line Maintenance ($000)"]-y_hat

print("The simple linear regression residual: {}".format(residual))

# adding a quadratic term to the linear model
X_squared = np.square(df_maint_cost["Customers (000's)"])
df_maint_cost['Customers Squared'] = X_squared
X = df_maint_cost[["Customers (000's)", "Customers Squared"]]
Y = df_maint_cost['Line Maintenance ($000)']
X = sm.add_constant(X)
model3 = sm.OLS(Y,X)
results3 = model3.fit()
print("The multiple linear regression model with the quadratic variable")
print(results3.params)
print(results3.summary())

print("The correlation between the two variables (the customer and the customer squared) is {}".format(np.corrcoef(df_maint_cost["Customers (000's)"], df_maint_cost['Customers Squared'])[1,0]))

X = df_maint_cost["Customers (000's)"]
Y = df_maint_cost['Customers Squared']
X = sm.add_constant(X)
model4 = sm.OLS(Y,X)
results4 = model4.fit()

# Calculate the VIF of the new quadratic variable
R_squared_of_quadratic = results4.rsquared
VIF = 1/(1-R_squared_of_quadratic)
print("the VIF of the two variables (the customer and the customer squared):{}".format(VIF))

# transform the data using the “centered predictor” method
X = df_maint_cost[["Customers (000's)", "Customers Squared"]]
Customers_transformed = X["Customers (000's)"]-np.mean(X["Customers (000's)"])
X["Customers (000's)"] = Customers_transformed
Customer_quadratic_transformed = np.square(Customers_transformed)
X["Customers Squared"] = Customer_quadratic_transformed
X.columns = ["Customers (000's) transformed", "Customers Squared transformed"]
X_transformed = X
Y = df_maint_cost['Line Maintenance ($000)']
X = sm.add_constant(X)
model5 = sm.OLS(Y,X)
results5 = model5.fit()
print("The multiple linear model using the transformed predictors")
print(results5.params)
print(results5.summary())

# Calculate the VIF of the new transformed quadratic variable
X = Customers_transformed
Y = Customer_quadratic_transformed
X = sm.add_constant(X)
model6 = sm.OLS(Y,X)
results6 = model6.fit()

R_squared_of__transformed_quadratic = results6.rsquared
VIF_transformed = 1/(1-R_squared_of__transformed_quadratic)
print("the VIF of the two transfomred variables (the customer and the customer squared):{}".format(VIF_transformed))

# calculate a 95% confidence interval for the predicted line maintenance costs for a utility serving 100,000 customers
MSres = results5.mse_resid
X0 = [100-np.mean(df_maint_cost["Customers (000's)"]), np.square(100-np.mean(df_maint_cost["Customers (000's)"]))]
X = X_transformed[["Customers (000's) transformed", "Customers Squared transformed"]]
slope1 = 14.4162
slope2 = 0.1543 
intercept = 955.6563
n = 12
k = 1
t_value = 2.228
y_hat_x_equal_100 = X0[0]*slope1 + X0[1]*slope2 + intercept

interval_with_95_confidence_when_x_equals_100 = [y_hat_x_equal_100 - t_value*np.sqrt(MSres*(1+int(np.dot(np.dot(np.transpose(X0),(np.matrix((np.dot(np.array(np.transpose(X)),np.array(X)))).I)),X0)))), 
                                                y_hat_x_equal_100 + t_value*np.sqrt(MSres*(1+int(np.dot(np.dot(np.transpose(X0),(np.matrix((np.dot(np.array(np.transpose(X)),np.array(X)))).I)),X0))))]

print("The 95% confidence interval for a prediction of maintenance costs for a utility serving 100,000 customers is: {}".format(interval_with_95_confidence_when_x_equals_100))
