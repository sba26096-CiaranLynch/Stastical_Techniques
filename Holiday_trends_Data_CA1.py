#!/usr/bin/env python
# coding: utf-8

# # Recent Holiday Trends and Traveller Behaviour Survey
# 

# ## The purpose of this survey is to explore recent holiday trends and traveller behaviour. This survey forms part of an academic project for the Statistical Techniques for Data Analytics module.

# ## Add python libraries to analysis

# In[25]:


import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np


# ## Data Preparation

# ## Load the Dataset

# In[26]:


df_Holidays = pd.read_csv("Holiday_trends_Data_CA1.csv")


# ### Display first 5 rows, to ensure the data has correctly loaded
# 

# In[27]:


df_Holidays.head(5)


# ###  Trim any spaces from the headings

# In[28]:


df_Holidays.columns = df_Holidays.columns.str.strip()


# ### Check for missing values

# In[29]:


df_Holidays.isnull().sum()


# ### No missing values were identified in the dataset. Therefore no records required removal

# In[30]:


df_Holidays.shape


# In[31]:


df_Holidays.info()


# # Rename columns to make it more readable 

# In[32]:


df_Holidays = df_Holidays.rename(columns={
    "Timestamp": "Timestamp",
    "What age range are you in?": "Age_Range",
    "Was your holiday": "Holiday_Location",
    "What was the main purpose of your Holiday": "Holiday_Purpose",
    "How did you book your holiday": "Booking_Method",
    "How recent was this holiday": "Holiday_Recency",
    "How would you describe your holiday type": "Holiday_Type",
    "How many people were in your travel group?": "Group_Size",
    "Approximately how much did your accommodation cost per room per night (€)?": "Accommodation_Cost",
    "How many nights did your holiday last?": "Holiday_Length",
    "Overall Holiday Cost - Approximate total spend for all travellers (€), including  cost of holiday + daily spending.": "Total_Cost",
    "How satisfied were you with your holiday overall?": "Satisfaction",
    "Would You Take This Type of Holiday Again?": "Holiday_Again",
    "Did You Travel by Air?": "Travel_By_Air",
    "How many holidays have you taken in the last 12 months?": "Holidays_Last_12_Months"
    } 
    )


# In[33]:


df_Holidays.head(5)


# ## Show Statistics of the dataset

# ###  Descriptive Statistics

# In[34]:


df_Holidays.describe().T


# ### Show Skewness of the data

# In[35]:


df_Holidays.skew(numeric_only=True)


# ### Accomodation Costs shows a positive skewness (6.33), indicating the presence of extreme observations. In Descriptive Statistics, the average accomodation cost is €221.10, the maximum recorded value is €4,000, suggesting a small number of very expensive holidays are influencing the overall distribution

# ### Holiday Length also demonstrates a positive skewness of 2.47.  While most holidays show an average of 7.5 days, the maximum holiday length of 28 days was recorded, again this outlier is influencing the overall distribution

# ### The Satisfaction skewness of (-0.74) indicates that responses are clustered towards the higher end of the rating scale.  This is suggesting that people were generally satisfied with their holiday experience

# In[36]:


df_Holidays.median(numeric_only=True)


# ### See above median values.  Accomodation costs has a median of €112.50, however the mean is €221.10.  This further demonstrates the impact of extreme values on the distribution

# ### Show counts of dataset for Age Range, Holiday Purpose and Booking Method

# In[37]:


df_Holidays['Age_Range'].value_counts()


# ### The largest responsent group was in the 25-34 range (12).  This was closely followed by the 35-44 and 45-54 age groups. The survey seems to have achieved a representation across most age categories, although fewer replies came from the over 65 age range

# In[38]:


df_Holidays['Holiday_Purpose'].value_counts()


# ### Beach holidays abroad were the most popular holiday type.  This is not surprising as the survey took place immediately after the summer holiday season.  However holidays to attend an event were the least common with 4 responses, again not surprising given the date of the survey

# ### Show Booking methods as a Percentage

# In[39]:


df_Holidays['Booking_Method'].value_counts(normalize=True) * 100


# ###  This insight shows a preference for booking holidays independently with 75% booking online as opposed to a package option.

# ## Run a pivot table to show Booking Methods broken down by Age Range

# In[40]:


# create a pivot table
pivot_bookings = pd.pivot_table (
    df_Holidays,
    values="Group_Size",
    index="Booking_Method",
    columns="Age_Range",
    aggfunc="count",
    margins=True,
    margins_name="Total"
)
pivot_bookings


# ## Independent online booking was the preferred booking method across all age groups. The highest level of independent booking was observed in the 25-34 age category. While package holidays became relatively more common amongst respondents aged 45-64, independently booking online remained the dominant method across every age group surveyed.

# ## Show the Booking Methods Pivot table as a bar chart

# In[41]:


pivot_chart = pivot_bookings.drop(
    index="Total",
    errors="ignore").drop(
    columns="Total",
    errors="ignore")

pivot_chart.plot(
    kind="bar",
    figsize=(10,6))

plt.title("Booking Methods by Age Range")
plt.xlabel("Booking Method")
plt.ylabel("Number of Respondents")
plt.legend(title="Age Range")

plt.show()


# ## Analysis of Categorical Variables

# In[47]:


df_age = df_Holidays['Age_Range'].value_counts()
df_age


# ### Reindex the data to show in age range order

# In[48]:


df_age = df_age.reindex( 
    ["25-34", "35-44", "45-54", "55-64", "65+"]
)


# ## Show Age Range Distribution in a Bar Chart

# In[49]:


plt.figure(figsize=(8,5) )

df_age.plot.bar(
    color="steelblue",
    edgecolor="black")

plt.title("Age Distribution of Respondents")
plt.xlabel("Age Range")
plt.ylabel("Number of Respondents")

plt.show()



# ## The age distribution chart shows that respondents were reasonably evenly distributed across the younger and middle-age categories. The 25–34 and 35–44 age groups represented the largest proportion of respondents, while the 65+ category contained the fewest responses.

# ### Create a new Data Frame called GroupSize.  Take columns, Group_Size, Holiday_Length, Total_Cost & Satisfaction.  

# ### A new derived variable, Cost_Per_Night, was created by dividing Total_Cost by Holiday_Length. This gives a new metric for exploratory analysis, allowing holiday spending to be standarsised across all durations

# In[87]:


df_Daily_Cost = df_Holidays[
    ["Age_Range","Group_Size", "Holiday_Length", "Total_Cost", "Satisfaction"] 
    ].copy()


# In[88]:


df_Daily_Cost.head(3)


# In[89]:


df_Daily_Cost["Cost_Per_Night"] = (
    df_Daily_Cost["Total_Cost"] / 
    df_Daily_Cost["Holiday_Length"] 
)
df_Daily_Cost.head(3)


# In[90]:


df_Daily_Cost.describe().T


# ### Create a Pivot Table on the mean off the cost per night and compare with age range.  Plot the results on a Bar chart

# In[94]:


pivot_Group = pd.pivot_table(
    df_Daily_Cost,
    values= "Cost_Per_Night",
    index="Age_Range",
    aggfunc="mean"
)

pivot_Group


# In[96]:


plt.figure(figsize=(8,5) )

pivot_Group.plot.bar(
    color="steelblue",
    edgecolor="black")

plt.title("Average Cost Per Nght by Age Range")
plt.xlabel("Age Range")
plt.ylabel("Average Cost (€)")


plt.show()


# #### The chart above shows that respondents aged 45-54 recorded the highest average cost per night (€451.94).
# #### This was followed by respondents aged 55-64 (€405.25) and 35-44(€369.99).
# #### Respondents aged 65+ recorded the llowest cost per night (€238.09).
# #### This suggests that middle aged respondents tend to spend more per night on the overall holiday experience, that theie younger or elder respondents.
# #### Although there is a noticeable decline in overall spending within the 65+ category, we need to consider the smaller sample size

# In[99]:


plt.figure(figsize=(8,6))

sns.heatmap(
df_GroupSize.corr(numeric_only=True),
annot=True,
cmap='coolwarm'
)
 
plt.title('Correlation Matrix')


# In[ ]:





# In[ ]:




