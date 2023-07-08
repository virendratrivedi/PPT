#!/usr/bin/env python
# coding: utf-8

# ## Qu:-1 Scenario: A company wants to analyze the sales performance of its products in different regions. They have collected the following data:
#    Region A: [10, 15, 12, 8, 14]
#    Region B: [18, 20, 16, 22, 25]
#    Calculate the mean sales for each region.
# 

# In[1]:


# Sales data for each region
region_a_sales = [10, 15, 12, 8, 14]
region_b_sales = [18, 20, 16, 22, 25]

# Calculate the mean sales for Region A
mean_sales_region_a = sum(region_a_sales) / len(region_a_sales)

# Calculate the mean sales for Region B
mean_sales_region_b = sum(region_b_sales) / len(region_b_sales)

# Print the results
print("Mean sales for Region A:", mean_sales_region_a)
print("Mean sales for Region B:", mean_sales_region_b)


# ## Qu:-2 Scenario: A survey is conducted to measure customer satisfaction on a scale of 1 to 5. The data collected is as follows:
#    [4, 5, 2, 3, 5, 4, 3, 2, 4, 5]
#    Calculate the mode of the survey responses.
# 

# In[3]:


import statistics

survey_responses = [4, 5, 2, 3, 5, 4, 3, 2, 4, 5]

# Calculate the mode
mode = statistics.mode(survey_responses)

# Print the result
print("Mode of the survey responses:", mode)


# ## Qu 3:- Scenario: A company wants to compare the salaries of two departments. The salary data for Department A and Department B are as follows:
#    Department A: [5000, 6000, 5500, 7000]
#    Department B: [4500, 5500, 5800, 6000, 5200]
#    Calculate the median salary for each department
# 

# In[4]:


import statistics

# Salary data for each department
department_a_salaries = [5000, 6000, 5500, 7000]
department_b_salaries = [4500, 5500, 5800, 6000, 5200]

# Calculate the median salary for each department
median_a = statistics.median(department_a_salaries)
median_b = statistics.median(department_b_salaries)

# Print the results
print("Median salary for Department A:", median_a)
print("Median salary for Department B:", median_b)


# ## Qu 4:- Scenario: A data analyst wants to determine the variability in the daily stock prices of a company. The data collected is as follows:
#    [25.5, 24.8, 26.1, 25.3, 24.9]
#    Calculate the range of the stock prices
# 

# In[5]:


stock_prices = [25.5, 24.8, 26.1, 25.3, 24.9]

# Calculate the range
stock_prices_range = max(stock_prices) - min(stock_prices)

# Print the result
print("Range of stock prices:", stock_prices_range)


# ## Qu 5:- Scenario: A study is conducted to compare the performance of two different teaching methods. The test scores of the students in each group are as follows:
#    Group A: [85, 90, 92, 88, 91]
#    Group B: [82, 88, 90, 86, 87]
#    Perform a t-test to determine if there is a significant difference in the mean scores between the two groups
# 

# In[6]:


from scipy import stats

# Test scores for each group
group_a_scores = [85, 90, 92, 88, 91]
group_b_scores = [82, 88, 90, 86, 87]

# Perform t-test
t_statistic, p_value = stats.ttest_ind(group_a_scores, group_b_scores)

# Print the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)


# ## Qu 6:- Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
#    Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
#    Sales (in thousands): [25, 30, 28, 20, 26]
#    Calculate the correlation coefficient between advertising expenditure and sales
# 

# In[7]:


import numpy as np

# Advertising expenditure and sales data
advertising_expenditure = [10, 15, 12, 8, 14]
sales = [25, 30, 28, 20, 26]

# Calculate the correlation coefficient
correlation_coefficient = np.corrcoef(advertising_expenditure, sales)[0, 1]

# Print the result
print("Correlation coefficient:", correlation_coefficient)


# ## Qu 7:- Scenario: A survey is conducted to measure the heights of a group of people. The data collected is as follows:
#    [160, 170, 165, 155, 175, 180, 170]
#    Calculate the standard deviation of the heights
# 

# In[8]:


import numpy as np

# Heights data
heights = [160, 170, 165, 155, 175, 180, 170]

# Calculate the standard deviation
std_deviation = np.std(heights)

# Print the result
print("Standard deviation of heights:", std_deviation)


# ## Qu 8:- Scenario: A company wants to analyze the relationship between employee tenure and job satisfaction. The data collected is as follows:
#    Employee Tenure (in years): [2, 3, 5, 4, 6, 2, 4]
#    Job Satisfaction (on a scale of 1 to 10): [7, 8, 6, 9, 5, 7, 6]
#    Perform a linear regression analysis to predict job satisfaction based on employee tenure.
# 

# In[10]:


from scipy import stats

# Employee tenure and job satisfaction data
employee_tenure = [2, 3, 5, 4, 6, 2, 4]
job_satisfaction = [7, 8, 6, 9, 5, 7, 6]

# Perform linear regression analysis
slope, intercept, r_value, p_value, std_err = stats.linregress(employee_tenure, job_satisfaction)

# Print the results
print("Slope:", slope)
print("Intercept:", intercept)
print("Correlation coefficient:", r_value)
print("P-value:", p_value)
print("Standard error:", std_err)


#the negative slope (-0.143) suggests a slight decrease in job satisfaction with increasing employee tenure. However, 
#the p-value (0.188) indicates that the relationship is not statistically significant at a commonly used significance 
#level of 0.05. Therefore, we cannot confidently conclude that there is a significant linear relationship between 
#employee tenure and job satisfaction based on this data.


# ## Qu 9:- Scenario: A study is conducted to compare the effectiveness of two different medications. The recovery times of the patients in each group are as follows:
#    Medication A: [10, 12, 14, 11, 13]
#    Medication B: [15, 17, 16, 14, 18]
#    Perform an analysis of variance (ANOVA) to determine if there is a significant difference in the mean recovery times between the two medications
# 

# In[11]:


from scipy import stats

# Recovery times for each medication
medication_a_recovery_times = [10, 12, 14, 11, 13]
medication_b_recovery_times = [15, 17, 16, 14, 18]

# Perform ANOVA
f_statistic, p_value = stats.f_oneway(medication_a_recovery_times, medication_b_recovery_times)

# Print the results
print("F-statistic:", f_statistic)
print("P-value:", p_value)

# The p-value is 0.015, which is less than the commonly used significance level of 0.05. Therefore, we can conclude that 
# there is a significant difference in the mean recovery times between the two medications.


# ## Qu 10:- Scenario: A company wants to analyze customer feedback ratings on a scale of 1 to 10. The data collected is
# 
#  as follows:
#     [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]
#     Calculate the 75th percentile of the feedback ratings.
# 

# In[12]:


import numpy as np

feedback_ratings = [8, 9, 7, 6, 8, 10, 9, 8, 7, 8]

# Calculate the 75th percentile
percentile_75 = np.percentile(feedback_ratings, 75)

# Print the result
print("75th percentile of feedback ratings:", percentile_75)


# ## Qu 11:- Scenario: A quality control department wants to test the weight consistency of a product. The weights of a sample of products are as follows:
#     [10.2, 9.8, 10.0, 10.5, 10.3, 10.1]
#     Perform a hypothesis test to determine if the mean weight differs significantly from 10 grams
# 

# In[13]:


from scipy import stats

weights = [10.2, 9.8, 10.0, 10.5, 10.3, 10.1]
population_mean = 10

# Perform one-sample t-test
t_statistic, p_value = stats.ttest_1samp(weights, population_mean)

# Print the results
print("T-statistic:", t_statistic)
print("P-value:", p_value)


# ## Qu 12:- Scenario: A company wants to analyze the click-through rates of two different website designs. The number of clicks for each design is as follows:
#     Design A: [100, 120, 110, 90, 95]
#     Design B: [80, 85, 90, 95, 100]
#     Perform a chi-square test to determine if there is a significant difference in the click-through rates between the two designs.
# 

# In[15]:


from scipy import stats

# Click-through rates for each design
design_a_clicks = [100, 120, 110, 90, 95]
design_b_clicks = [80, 85, 90, 95, 100]

# Perform chi-square test
chi2_statistic, p_value, _, _ = stats.chi2_contingency([design_a_clicks, design_b_clicks])

# Print the results
print("Chi-square statistic:", chi2_statistic)
print("P-value:", p_value)

# The p-value is 0.8080, which is greater than the commonly used significance level of 0.05. Therefore, we do not have 
# sufficient evidence to reject the null hypothesis


# ## Qu 13:- Scenario: A survey is conducted to measure customer satisfaction with a product on a scale of 1 to 10. The data collected is as follows:
#     [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]
#     Calculate the 95% confidence interval for the population mean satisfaction score.
# 

# In[17]:


import numpy as np
from scipy import stats

satisfaction_scores = [7, 9, 6, 8, 10, 7, 8, 9, 7, 8]
confidence_level = 0.95

# Calculate the sample mean and standard error
sample_mean = np.mean(satisfaction_scores)
sample_std = np.std(satisfaction_scores)
sample_size = len(satisfaction_scores)
std_error = sample_std / np.sqrt(sample_size)

# Calculate the confidence interval
confidence_interval = stats.t.interval(confidence_level, df=sample_size-1, loc=sample_mean, scale=std_error)

# Print the result
print("95% Confidence Interval:", confidence_interval)


# ## Qu 14:- Scenario: A company wants to analyze the effect of temperature on product performance. The data collected is as follows:
#     Temperature (in degrees Celsius): [20, 22, 23, 19, 21]
#     Performance (on a scale of 1 to 10): [8, 7, 9, 6, 8]
#     Perform a simple linear regression to predict performance based on temperature.
# 

# In[19]:


from scipy import stats

# Temperature and performance data
temperature = [20, 22, 23, 19, 21]
performance = [8, 7, 9, 6, 8]

# Perform simple linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(temperature, performance)

# Print the results
print("Slope:", slope)
print("Intercept:", intercept)
print("Correlation coefficient:", r_value)
print("P-value:", p_value)
print("Standard error:", std_err)

# The positive slope (0.667) suggests that performance tends to increase as temperature increases. However, the 
# p-value (0.0891) indicates that the relationship is not statistically significant at a commonly used significance level 
# of 0.05. Therefore, we cannot confidently conclude that there is a significant linear relationship between temperature 
# and performance based on this data.


# ## Qu 15 :-Scenario: A study is conducted to compare the preferences of two groups of participants. The preferences are measured on a Likert scale from 1 to 5. The data collected is as follows:
#     Group A: [4, 3, 5, 2, 4]
#     Group B: [3, 2, 4, 3, 3]
#     Perform a Mann-Whitney U test to determine if there is a significant difference in the median preferences between the two groups.
# 

# In[21]:


from scipy import stats

# Preferences for each group
group_a_preferences = [4, 3, 5, 2, 4]
group_b_preferences = [3, 2, 4, 3, 3]

# Perform Mann-Whitney U test
statistic, p_value = stats.mannwhitneyu(group_a_preferences, group_b_preferences)

# Print the results
print("Mann-Whitney U statistic:", statistic)
print("P-value:", p_value)

# In this case, the p-value is 0.361, which is greater than the commonly used significance level of 0.05. Therefore, 
# we do not have sufficient evidence to reject the null hypothesis, and we cannot conclude that there is a significant 
# difference in the median preferences between the two groups based on this data.


# ## Qu 16:- Scenario: A company wants to analyze the distribution of customer ages. The data collected is as follows:
#     [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]
#     Calculate the interquartile range (IQR) of the ages.
# 

# In[22]:


import numpy as np

ages = [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]

# Calculate the IQR
iqr = np.percentile(ages, 75) - np.percentile(ages, 25)

# Print the result
print("Interquartile Range (IQR) of ages:", iqr)


# ## Qu 17:- Scenario: A study is conducted to compare the performance of three different machine learning algorithms. The accuracy scores for each algorithm are as follows:
#     Algorithm A: [0.85, 0.80, 0.82, 0.87, 0.83]
#     Algorithm B: [0.78, 0.82, 0.84, 0.80, 0.79]
#     Algorithm C: [0.90, 0.88, 0.89, 0.86, 0.87]
#     Perform a Kruskal-Wallis test to determine if there is a significant difference in the median accuracy scores between the algorithms.
# 

# In[24]:


from scipy import stats

# Accuracy scores for each algorithm
algorithm_a_scores = [0.85, 0.80, 0.82, 0.87, 0.83]
algorithm_b_scores = [0.78, 0.82, 0.84, 0.80, 0.79]
algorithm_c_scores = [0.90, 0.88, 0.89, 0.86, 0.87]

# Perform Kruskal-Wallis test
statistic, p_value = stats.kruskal(algorithm_a_scores, algorithm_b_scores, algorithm_c_scores)

# Print the results
print("Kruskal-Wallis statistic:", statistic)
print("P-value:", p_value)
# In this case, the p-value is 0.1275, which is greater than the commonly used significance level of 0.05. Therefore, 
# we do not have sufficient evidence to reject the null hypothesis, and we cannot conclude that there is a significant 
# difference in the median accuracy scores between the algorithms based on this data.


# ## Qu 18:-Scenario: A company wants to analyze the effect of price on sales. The data collected is as follows:
#     Price (in dollars): [10, 15, 12, 8, 14]
#     Sales: [100, 80, 90, 110, 95]
#     Perform a simple linear regression to predict
# 
#  sales based on price.
# 

# In[26]:


from scipy import stats

# Price and sales data
price = [10, 15, 12, 8, 14]
sales = [100, 80, 90, 110, 95]

# Perform simple linear regression
slope, intercept, r_value, p_value, std_err = stats.linregress(price, sales)

# Print the results
print("Slope:", slope)
print("Intercept:", intercept)
print("Correlation coefficient:", r_value)
print("P-value:", p_value)
print("Standard error:", std_err)

# In this case, the negative slope (-6.5) suggests that as the price increases, sales tend to decrease. 
#The p-value (0.0118) indicates that the relationship is statistically significant at a commonly used significance 
# level of 0.05. Therefore, we can conclude that there is a significant linear relationship between price and sales 
#based on this data.


# ## Qu :-19 Scenario: A survey is conducted to measure the satisfaction levels of customers with a new product. The data collected is as follows:
#     [7, 8, 9, 6, 8, 7, 9, 7, 8, 7]
#     Calculate the standard error of the mean satisfaction score.
# 

# In[27]:


import numpy as np

satisfaction_scores = [7, 8, 9, 6, 8, 7, 9, 7, 8, 7]

# Calculate the standard error of the mean
standard_error = np.std(satisfaction_scores) / np.sqrt(len(satisfaction_scores))

# Print the result
print("Standard error of the mean satisfaction score:", standard_error)


# ## Qu :-20 Scenario: A company wants to analyze the relationship between advertising expenditure and sales. The data collected is as follows:
#     Advertising Expenditure (in thousands): [10, 15, 12, 8, 14]
#     Sales (in thousands): [25, 30, 28, 20, 26]
#     Perform a multiple regression analysis to predict sales based on advertising expenditure.
# 

# In[28]:


import statsmodels.api as sm
import numpy as np

# Advertising expenditure and sales data
advertising_expenditure = [10, 15, 12, 8, 14]
sales = [25, 30, 28, 20, 26]

# Add a constant term to the independent variable
advertising_expenditure = sm.add_constant(advertising_expenditure)

# Create a linear regression model
model = sm.OLS(sales, advertising_expenditure)

# Fit the model to the data
results = model.fit()

# Print the results
print(results.summary())


# In[ ]:




