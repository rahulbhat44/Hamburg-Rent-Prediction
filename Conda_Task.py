#!/usr/bin/env python
# coding: utf-8

# # Condo Rent Prediction Task - Geo-Data Enrichment
#
# The objective of the task is to build a predictive model using machine learning models that would be able to predict a rent per square meter in Hamburg city.
#
# <br>
# Additional geo-data information or features has to be fetched from the external sources and concatenate with the dataset given. The external features could be anything useful for the analysis such as places nearby, bus stops, railway stations etc.
#
# ## Dataset Description
#
# Number of variables: 15
# <br>
# Number of observations: 12500
# <br>
# - living_space: The total area of the rented apartment
# <br>
# - rooms: Number of rooms availabe for rent
# <br>
# - cold_rent - The monthly rent price
# <br>
# - construction_year - The year in which the apartment is builtquarter - The
# <br>
# - quarter - A town or district in the Hmaburg city
# <br>
# - lambert_poistion_x - Map projection or location of the apartment
# <br>
# - lambert_poistion_y - Map projection or location of the apartment
# <br>
# - city - Name of the city
# <br>
# - postcode - The postal code of the area in which the apartment is available for rent
# <br>
# - heating_type - What type of heating is avilable in the apartment
# <br>
# - number_of_bedrooms - The number of bedrooms
# <br>
# - rent_per_square_meter - The rent in square meters
# <br>
# - publish_date - The date on which it is published
# <br>
# - latitude - The geographic coordinates of the apartment
# <br>
# - longitude - The geographic coordinates of the apartment
# <br>
#
# ## Data Profiling
#
# Data profiling is also done using the pandas data profiling, by using this we can automatically genetrate the profile reports from a pandas DataFrame. The pandas df.describe() function is great but a little basic for serious exploratory data analysis. pandas_profiling extends the pandas DataFrame with df.profile_report() and does the exploratory analysis for us. It is fast and efficient. This helps to understand the data very well. The report is also present in the folder by the name of data_profile.
#
# ## Libraries
#
# If you need all the libraries I have used for this task, please remove the hashtags and install it directly from this script for example "!{sys.executable} -m pip install geopandas". You have to just run it and it will download the library you need

# In[1]:


import sys
# !{sys.executable} -m pip install geopandas
# !{sys.executable} -m pip install mplleaflet
# !{sys.executable} -m pip install bs4
# !{sys.executable} -m pip install geocoder
# !{sys.executable} -m pip install geopy
# !{sys.executable} -m pip install folium
# !{sys.executable} -m pip install lxml
# !{sys.executable} -m pip install pygeoj
# !{sys.executable} -m pip install pyshp
# !{sys.executable} -m pip install datetime
# !{sys.executable} -m pip install seaborn
# !{sys.executable} -m pip install --upgrade cython


# In[2]:


# required package for timeseries
# !{sys.executable} -m pip install statsmodels


# In[3]:


# # required package to run geopandas
#!conda install -c conda-forge libspatialindex -y


# In[4]:


# # required packages for neighbourhood analysis
# !{sys.executable} -m pip install geopandas
# !{sys.executable} -m pip install descartes
# !{sys.executable} -m pip install requests


# In[5]:


# # requiered packages for accessibility analysis
# # Make sure Cython is upgraded FIRST!
# !{sys.executable} -m pip install pandana


# In[6]:


# requiered packages for modelling
#!{sys.executable} -m pip install xgboost


# In[7]:


get_ipython().run_line_magic('matplotlib', 'inline')

# general
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Geographical analysis
import geopandas as gpf #libspatialindex nees to be installed first
import json # library to handle JSON files
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe
import geopandas as gpd
import shapefile as shp
import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
import requests
import descartes
import rtree

# accessibility analysis
import time
# from pandana.loaders import osm
# from pandana.loaders import pandash5

# modelling
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn import metrics
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import explained_variance_score, mean_squared_error, r2_score

#Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Set plot preference
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

print('Libraries imported.')


# ### Libraries for pandas data profiling which does the exploratory analysis.

# In[8]:


from pathlib import Path

# Installed packages
from ipywidgets import widgets

# Our package
from pandas_profiling import ProfileReport
from pandas_profiling.utils.cache import cache_file


# ### Libraries for plotly graphs

# In[9]:


# import plotly.plotly as py
# import plotly.graph_objs as go
# from plotly.offline import iplot, init_notebook_mode
from chart_studio.plotly import plot, iplot
import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


#  # 1. Cleaning and Pre-processing

# In[10]:


# Load the dataset
df = pd.read_csv('task_data_final.csv')
print(f"The dataset contains {len(df)} Condo Rent listings")
pd.set_option('display.max_columns', len(df.columns)) # To view all columns
pd.set_option('display.max_rows', 100)
df.head(10)


# In[11]:


df.count()


# In[12]:


df.head()


# In[13]:


df.tail()


# # 2. Data Profiling

# In[14]:


profile = ProfileReport(df, title="Condo Rent Prices", html={'style': {'full_width': True}}, sort="None")


# In[15]:


# Or use the HTML report in an iframe
profile.to_notebook_iframe()


# In[16]:


profile.to_file(output_file="data_profile.html")


# In[17]:


df.describe()


# From the above summary we can notice one thing in rooms column and that is there is no 1 room apartment available for rent, the minimum number of rooms are 1.5. The maximum number of rooms are 9.
#
# The minimum cold rent is 159.60 and the maximum is 4788.00. The minimum rent per suare meter is 6.00 and the maximum is 35.49.

# In[18]:


# Checking the null values in a dataset
df.isnull().any()


# In[19]:


# Sum of null values
df.isnull().sum()


# There are many missing values in the dataset especially the number_of_bedrooms i.e 7722. It would be better to get rid of the columns with so many missing values.

# In[20]:


def percent_value_counts(df, feature):
    """This function takes in a dataframe and a column and finds the percentage of the value_counts"""
    percent = pd.DataFrame(round(df.loc[:,feature].value_counts(dropna=False, normalize=True)*100,2))
    ## creating a df with the
    total = pd.DataFrame(df.loc[:,feature].value_counts(dropna=False))

    ## concating percent and total dataframe
    total.columns = ["Total"]
    percent.columns = ['Percent']
    return pd.concat([total, percent], axis = 1)


# In[21]:


percent_value_counts(df, 'number_of_bedrooms')


# In[22]:


percent_value_counts(df, 'construction_year')


# We can see the total number of missing values and percentage of missing values for a particular attribute.

# In[23]:


# Plotting the distribution of numerical and boolean categories
df.hist(figsize=(20,20));


# From the above, it can be seen that several columns only contain few category and can be dropped such as number_of_bedrooms, construction_year, heating_type. We can also remove lambert position, we don't need that. Later we can see what else we can remove.

# In[24]:


df.drop(['number_of_bedrooms', 'construction_year', 'heating_type', 'lambert_poistion_x', 'lambert_poistion_y'], axis=1, inplace=True)


# In[25]:


df.isna().sum()


# Now still we have missing values, we need the locations of the apartments and 80 are missing, we can get rid of that.

# In[26]:


df = df.dropna()


# In[27]:


df.isna().sum()


# In[28]:


df.count()


# Now the dataset is clean and perfect for the analysis and prediction.

# # 3. Analysis and Visualization

# Let's start with the bar graph, let's see the cold rent according to the rooms. It is but obvious that if the number of rooms increase the rent also increases. Let's check that.

# In[29]:


rent_pivot = df.pivot_table(index="rooms",values="cold_rent")
rent_pivot.plot.bar()
plt.show()


# In the above bar graph we can clearly see that as the number of rooms increases, the cold rent increases. But in case of 3.7 rooms (don't know if there is 2.7 or 3.7 rooms available) the rent is low as compare to others.
# Let's check for the rent per square meter with rooms. Is there any difference?

# In[30]:


rent_pivot = df.pivot_table(index="rooms",values="rent_per_square_meter")
rent_pivot.plot.bar()
plt.show()


# In the above plot we can again see that 3.7 rooms has lower rent price as compare to others. We can also replace 2.7 and 3.7 with 2.5 and 3.5 but we will keep it like this. The rent per square meter bar graph looks good, 2.0 and 2.5 rooms per square meter price is lower than the 1.5 room.
#
# I am very curious about the 2.7 and 3.7 rooms. Let's check how many 2.7 and 3.7 rooms are there in a dataset.

# In[31]:


percent_value_counts(df, 'rooms')


# Now we can clearly see that there are total of 7 and 0.06% of 2.7 rooms and the total of 2 and 0.02% of 3.7 rooms present in the dataset. I believe this is some typing error or mistake. We can simply replace 2.7 rooms with 2.5 and 3.7 rooms with 3.5 as round numbers.

# In[32]:


# replacing the 2.7 and 3.7 rooms with 2.5 and 3.5
df.rooms= df.rooms.astype(str).str.replace('3.7','3.5',regex=True)
df.rooms= df.rooms.astype(str).str.replace('2.7','2.5',regex=True)


# In[33]:


# Checking the replacement
percent_value_counts(df, 'rooms')


# In[34]:


# Checking the rent per square meter in the city
pd.set_option('display.max_rows', 500)
df.groupby('city').max()['rent_per_square_meter']


# There are so many spelling mistakes, syntax and errors in the column city name. It is not clean and properly
# managed the names. It could give us the wrong impression and data values. It is not also good for the prediction model. We need to get rid of these spelling mistakes and syntax. Somewhere it is written Hamnurg, Hammburg etc. Somewhere it is like Altona-Altstadt and the same is Hamburg Altona-Altstadt. We need to clean this cloumn.

# In[35]:


df = df[df.city != 'H']
df = df[df.city != 'hgttfrf']
df = df[df.city != 'osdorf']
df.city= df.city.str.replace('Hamburg/','Hamburg ',regex=True)
df.city= df.city.str.replace('Hamburg /','Hamburg ',regex=True)
df.city= df.city.str.replace('Hamburg-','Hamburg ' ,regex=True)
df.city= df.city.str.replace('Hamburg -','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg,','Hamburg ',regex=True)
df.city= df.city.str.replace('HAMBURG -','Hamburg',regex=True)
df.city= df.city.str.replace('Hambrug','Hamburg ',regex=True)
df.city= df.city.str.replace('Hambur','Hamburg ',regex=True)
df.city= df.city.str.replace('Ha,mburg','Hamburg ',regex=True)
df.city= df.city.str.replace('Hamburghttps://www.immobilienscout24.de/scoutmanag','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg https://www.immobilienscout24.de/scoutmanag','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburghttps://www.immobilienscout24.de/scoutmanag','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg g','Hamburg ',regex=True)
df.city= df.city.str.replace('Hamburg t','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg ug','Hamburg',regex=True)
df.city= df.city.str.replace('Hanburg','Hamburg',regex=True)
df.city= df.city.str.replace('Harburg','Hamburg',regex=True)
df.city= df.city.str.replace('HAMBURG','Hamburg',regex=True)
df.city= df.city.str.replace('Hmaburg','Hamburg',regex=True)
df.city= df.city.str.replace('20535 hamburg','Hamburg',regex=True)
df.city= df.city.str.replace('Mitte - Hamburg  Billstedt','Hamburg Billstedt',regex=True)
df.city= df.city.str.replace('22455','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg ','Hamburg',regex=True)
df.city= df.city.str.replace('-Bramfeld','',regex=True)
df.city= df.city.str.replace('Hamnurg ','Hamburg',regex=True)
df.city= df.city.str.replace('Hamnurg','Hamburg',regex=True)
df.city= df.city.str.replace('Hammburg','Hamburg',regex=True)
df.city= df.city.str.replace('Hamrburg','Hamburg',regex=True)
df.city= df.city.str.replace('hamburg-','Hamburg ',regex=True)
df.city= df.city.str.replace('hamburg','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburghttps://www.immobilienscout24.de/scoutmanag','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg Hamburg','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg  HafenCity, Osakaallee 6','Hamburg',regex=True)
#df.city= df.city.str.replace('Altona-Nord','Hamburg Altona-Nord',regex=True)
df.city= df.city.str.replace('Altona-Altstadt','Hamburg Altona-Altstadt',regex=True)
df.city= df.city.str.replace('Hamburg/ Wilstorf','Hamburg Wilstorf',regex=True)
df.city= df.city.str.replace('Bramfeld, Hamburg','Hamburg',regex=True)
df.city= df.city.str.replace('Hamburg  ','Hamburg ',regex=True)



# In[36]:


# df['city'] = df['city'].str.strip()
# df['city'] = df['city'].str.lstrip()
# df['city'] = df['city'].str.rstrip()
# #df['city'] = df['city'].str.replace("  "," ")


# In[37]:


# lets check it again
pd.set_option('display.max_rows', 500)
df.groupby(('city'), sort=False)['rent_per_square_meter'].max()


# Now it looks much better than the previous one. We removed the empty spaces, hyphens, commas and cleaned some names in a proper manner or format. Hmaburg shows the high rent rates of 35.496 per square meter. There are few cities which has the higher rent in 30's. Let's see the same in a plot, that would be much clearer and visible than this.

# Now we will use the plotly graphs to build some beautiful and interactive charts, we are using plotly because
# there are so many categories and values and by using plotly we can easily zoom in or zoom out the graphs and
# whenever there is something which is not clear we can hover the mouse pointer to the graphs and check the values,
# names, numbers related to that. We can also select and deselect the attributes and check the values.

# In[38]:


df.pivot(columns='city', values='rent_per_square_meter').iplot(
        kind='box',
        yTitle='Rent Per Square Meter',
        title='City and Rent')


# Now in the above box plot graph if we move hover the mouse cursor to the graph it shows the values and we can also select and deselect the attributes on the right hand side. With the box plot above we can see the rent per square meter in the city. We can also check the minimum rent, mean and maimum rent in the city.
#
# We can see that maimum rent is in the city Hamburg and Hamburg Hafencity with 35.4 followed by Hamburg Altona-Nord with 35.196.
#
# The interesting city is Hamburg Altona-Altstadt, the apartment rents are very high there above 31. The maximum rent is 32.22, the median is 31.848 and the minimum is 31.476. Hamburg has the maximum rent but it also has the minimum rent of 6 per square meter as well.
#
# Hamburg Mümmelmannsberg has the maximum rent of 6.624, median 6.384 and minimum of 6.
#

# In[39]:


# Checking the rent per square meter in the quarter. We can also see if the names written are correct or in the
# right format
pd.set_option('display.max_rows', 500)
df.groupby(('quarter'), sort=False)['rent_per_square_meter'].max()


# Quarter names and everything looks fine, lets check the same with a box plot.

# In[40]:


df.pivot(columns='quarter', values='rent_per_square_meter').iplot(
        kind='box',
        yTitle='Rent Per Square Meter',
        title='Quarter and Rent')


# The maximum rent is 35.496 in the Marienthal quarter followed by hafencity with 35.4.
# Cranz has the maximum of 10.416, median 9.816 and minimum 8.952.

# In[41]:


# Group by city and quarter just to simply check the names.
pd.set_option('display.max_rows', 500)
df.groupby(('city'), sort=False)['quarter'].max()


# In[42]:


# rent per square meter with the poscode
pd.set_option('display.max_rows', 500)
df.groupby(('postcode'), sort=True)['rent_per_square_meter'].max()


# The maimum rent in the list is showing in the postcode 20457 with 35.400. We know that the maimum rent of 35.4 is
# in Hamburg or the Hafencity. Lets check the where this zipcode belongs to.

# In[43]:


pd.set_option('display.max_rows', 500)
df.groupby(('postcode'), sort=True)['city'].max()


# From the above list we can see that there is 2103 postcode of 4 digits which seems wrong. We will remove that. Postcode 20457 belongs to Hafencity. We dont know where this post code belongs to, the possibilities could be:
# 21031 - Hamburg Lohbrügge
# <br>
# 21033 - Hamburg Lohbrügge
# <br>
# 21035 - Hamburg Bergedorf
# <br>
# 21039 - Hamburg Vierlanden
# <br>
# We will remove this from the dataset.

# In[44]:


df = df[df.postcode != 2103]


# In[45]:


pd.set_option('display.max_rows', 500)
df.groupby(('postcode'), sort=True)['city'].max()


# Now its clean with names. We can save the cleaned data.

# In[46]:


# Save cleaned dataset
cleaned_data = df.to_csv(r'rent_cleaned.csv', index=id, header=True)


# In[47]:


# Read dataset
df = pd.read_csv('rent_cleaned.csv', index_col=0)
df.head()


# In[48]:


df.isnull().sum()


# let's do some analysis and plot some graphs to understand the dataset better. let's start with the rent price range and do some more analysis and

# In[49]:


print(f"Advertised prices range from €{min(df.rent_per_square_meter)} to €{max(df.rent_per_square_meter)}.")


# In[50]:


# Distribution of prices from €6.0 to €35.496
plt.figure(figsize=(20,4))
df.rent_per_square_meter.hist(bins=10, range=(0,100))
plt.margins(x=0)
plt.axvline(15, color='blue', linestyle='--')
plt.title("Rent Prices Per Square Meter", fontsize=16)
plt.xlabel("Price Per SQ Meter(€)")
plt.ylabel("Number of listings")
plt.show()


# In[51]:


print(f"Advertised prices range from €{min(df.cold_rent)} to €{max(df.cold_rent)}.")


# In[52]:


# Distribution of prices from €159.6 to €4788.0
plt.figure(figsize=(20,4))
df.cold_rent.hist(bins=100, range=(159,4788))
plt.margins(x=0)
plt.axvline(1000, color='blue', linestyle='--')
plt.title("Cold Rent Prices", fontsize=16)
plt.xlabel("Price Cold rent(€)")
plt.ylabel("Number of listings")
plt.show()


# In[53]:


# Distribution of prices between €1000 and €2000
plt.figure(figsize=(20,4))
df.cold_rent.hist(bins=100, range=(159,4788))
plt.margins(x=0)
plt.axvline(1000, color='green', linestyle='--')
plt.axvline(2000, color='red', linestyle='--')
plt.title("Cold Rent up to €1000", fontsize=16)
plt.xlabel("Price (€)")
plt.ylabel("Number of listings")
plt.show()


# We can see that there are more apartments for rent under 1000 Euros whereas less number of apartments for rent
# above 1000 Euros. After 2000 Euros there are few apartments left for the rent.

# # Which is the most common house (Bedroom wise) ?

# In[54]:


df['rooms'].value_counts().plot(kind='bar')
plt.title("Number of Rooms")
plt.xlabel("Rooms")
plt.ylabel("Count")


# As we can see from the visualization 2.5 bedroom houses are most common for the rent followed by 3.5 bedroom.

# In[55]:


df['quarter'].value_counts().iplot(
        kind='bar',
        yTitle='Most Apartments for Rent in which Quarter',
        title='Quarter')


# Winterhude has the highest number of apartments for the rent.

# # How common factors are affecting the price of the houses ?

# What factors affecting the prices of the house?
# Let us start with , If price is getting affecting by living area of the house or not ?

# In[56]:


plt.scatter(df.cold_rent, df.living_space)
plt.title("Cold Rent Vs Living Space")
plt.xlabel("Cold Rent")
plt.ylabel("Living SPace")
plt.show()


# The scatter plot above shows that the data points are in the linear direction and it clearly shows that if living area increases the price increases.

# In[57]:


plt.scatter(df.cold_rent, df.rooms)
plt.title("Rooms and Price")
plt.xlabel("Cold Rent")
plt.ylabel("Rooms")
plt.show()


# In[58]:


plt.scatter(df.postcode, df.cold_rent)
plt.title("price according to the location")


# From the above figure we can see that there are some wrong postcodes in the dataset as well. It looks like Hamburg postcode is between 20000 to 23000 but there are some wrong zipcodes.

# # Does published date affect the rent?

# In[59]:


df.pivot(columns='publish_date', values='rent_per_square_meter').iplot(
        kind='box',
        yTitle='Rent Per Square Meter',
        title='Published Dates and Rent')


# Yes it does look like newly published apartments are little higher than the old published dates. Before July 2016 until Jan 2017 the property rents are low but after July 2017 the rents are almost same. We could also do the same with the construction year but unfortunately we had to get rid of that because of so many missing values.
#

# In[60]:


df.pivot(columns='rooms', values='cold_rent').iplot(
        kind='box',
        yTitle='Rent Per Square Meter',
        title='Rooms and Rent')


# # 4. Geo-Data extract from Geofabrik
# "http://download.geofabrik.de/europe/germany/hamburg.html". This website is an open portal and has data extracts from the OpenStreetMap project which are normally updated every day. Select your continent and then your country of interest from the list below. This open data download service is offered free of charge by Geofabrik GmbH. I downloaded the shape file of venues for Hamburg and then below shaped file is read and converted to GeoJSON. We converted to GeoJSON to get the geometry out of the shape file.

# In[61]:


# Importing the shape file of the venues and converting to GeoJSON
import geopandas
myshpfile = geopandas.read_file('shape_file/gis_osm_pois_free_1.shp')
myshpfile.to_file('myJson.geojson', driver='GeoJSON')


# In[62]:


# Importing the city venues or places as a GeoJSON file as a dataframe in geopandas
df2 = gpd.read_file(r'myJson.geojson')
df2.head()


# In[63]:


# Check the missing values
df2.isna().sum()


# ## Now we need to concatenate the venue dataset and the rent dataset.  For that both datasets should have a point geometry. So, we will create a geometry for a rent dataset from its latitude and longitude

# In[64]:


df3 = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))


# In[65]:


df3.head()


# Now to concatenate the two datasets we need a nearest neighbour so that we can get the venues which are nearby to the apartment. I will use scipy's cKDTree spatial index.query method which returns very fast results for nearest neighbor searches. As it uses a spatial index it's orders of magnitude faster than looping though the dataframe and then finding the minimum of all distances. It is also faster than using shapely's nearest_points with RTree (the spatial index method available via geopandas) because cKDTree allows you to vectorize your search whereas the other method does not.
#
# Here is a helper function that will return the distance and 'Name' of the nearest neighbor from our external venues dataset to each point in our Condo Rent datatset. It assumes both datatsets have a geometry column of points.
#
# Not only we get the nearest venue or place to the apartment we will also get the distance in meteres as well. Later we can also convert the meters in Kilometers.

# In[66]:


from scipy.spatial import cKDTree
from shapely.geometry import Point


def ckdnearest(gdA, gdB):
    nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
    btree = cKDTree(nB)
    dist, idx = btree.query(nA, k=1)
    gdf = pd.concat(
        [gdA.reset_index(drop=True), gdB.loc[idx, gdB.columns != 'geometry'].reset_index(drop=True),
         pd.Series(dist, name='dist')], axis=1)
    return gdf

ckdnearest(df3, df2)


# In[67]:


ckdnearest(df3, df2).head()


# In[68]:


# Save the merged dataset
cleaned_merged_data = ckdnearest(df3, df2).to_csv(r'ckdnearest(df3, df2).csv', index=id, header=True)


# In[69]:


# Read dataset
df = pd.read_csv('ckdnearest(df3, df2).csv')


# In[70]:


df.head()


# In[75]:


df.drop(['Unnamed: 0'], axis=1, inplace=True)


# In[76]:


df.isna().sum()


# Only name column has empty data, we will keep this to check the names of the venues and later we can remove it at the time of prediction.

# ## We have successfully conactenated our datatset and also got the distance from the apartment to the venue as well. Now for the sake of simplicity we can convert the distance in meters into kilometers.

# In[77]:


# Convert Meters to Kilometers
df['dist'] = df['dist'].div(1000)


# In[78]:


df.head()


# Now we can see the distance is in Kilometers. We just simply divided the distance with kilometers. Now we can see the distance between the apartment and the nearby venue or place.

# # 5. Analysis of the city and the venue nearby

# In[79]:


df.groupby('quarter').count()


# In[80]:


print('There are {} unique categories.'.format(len(df['fclass'].unique())))


# In[81]:


df.groupby('fclass').count()


# In[82]:


# One Hot Encoding
hamburg_onehot = pd.get_dummies(df[['fclass']], prefix = "", prefix_sep = "")

## Add city column back to df
hamburg_onehot['quarter'] = df['quarter']

# Move neighbourhood column to the first column
fixed_columns = [hamburg_onehot.columns[-1]] + list(hamburg_onehot.columns[:-1])
hamburg_onehot = hamburg_onehot[fixed_columns]

# display
hamburg_onehot.head()


# In[83]:


hamburg_onehot.tail()


# In[84]:


# New df dimensions
hamburg_onehot.shape


# Group rows by neighbourhood and by taking the mean and the frequency of occurrence of each category

# In[85]:


hamburg_grouped = hamburg_onehot.groupby('quarter').mean().reset_index()
hamburg_grouped


# Get each neighbourhood along with its top 5 most common venues

# In[86]:


num_top_venues = 5

for hood in hamburg_grouped['quarter']:
    print("----"+hood+"----")
    temp = hamburg_grouped[hamburg_grouped['quarter'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')


# Put that in pandas data frame

# In[87]:


# Function to sort venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)

    return row_categories_sorted.index.values[0:num_top_venues]


# In[88]:


# New dataframe ordered
indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['quarter']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighbourhoods_venues_sorted = pd.DataFrame(columns=columns)
neighbourhoods_venues_sorted['quarter'] = hamburg_grouped['quarter']

for ind in np.arange(hamburg_grouped.shape[0]):
    neighbourhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(hamburg_grouped.iloc[ind, :],
                                                                          num_top_venues)

neighbourhoods_venues_sorted.head()


# In[89]:


neighbourhoods_venues_sorted.tail()


# Let's look at how many categories are on the 1st most common venue.

# In[90]:


print('There are {} unique categories.'.format(len(neighbourhoods_venues_sorted['1st Most Common Venue'].unique())))


# In[91]:


neighbourhoods_venues_sorted.groupby('1st Most Common Venue').count()


# From above, we observe that the most common venues across the dataset are Memorial, Post Box, Bench, Play Ground, Recycling Paper, followed by Restaurant. It is clear that different restaurant venues are in subcategories which makes them less common than if they were aggregated. Thus, for the purpose accounting for the venues that may have the most impact on price, we will limit the venues to the most common categories. It is unlikely that having a hotels, restaurants will affect price. I think these places or venues are not affecting the rent prices at all. Thus, that category of venues will not be considered.

# I was just cuirious to check whether the rent per square meter is same as cold rent / living space. As we can see it is same.

# # 6. Preparing data for modelling

# In[92]:


# Read dataset
df = pd.read_csv('ckdnearest(df3, df2).csv')
df.head()


# In[93]:


# Dropping variables no longer needed
df.drop(['longitude', 'latitude', 'name', 'osm_id', 'Unnamed: 0', 'geometry', 'code', 'fclass', 'name', 'dist', 'quarter', 'publish_date'], axis=1, inplace=True)


# In[94]:


df.head()


# We get dummies for our categorical variables to get the dataset ready for multicollinearity analysis.

# In[95]:


transformed_df = pd.get_dummies(df)
transformed_df.head()


# We now assess for multicollinearity of features:

# In[96]:


def multi_collinearity_heatmap(df, figsize=(11,9)):

    """
    Creates a heatmap of correlations between features in the df. A figure size can optionally be set.
    """

    # Set the style of the visualization
    sns.set(style="white")

    # Create a covariance matrix
    corr = df.corr()

    # Generate a mask the size of our covariance matrix
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, vmax=corr[corr != 1.0].max().max());


# In[97]:


multi_collinearity_heatmap(transformed_df, figsize=(40,40))


# It doesn't look like there are any significant collinear relationship with city, so these will temporarily be dropped to produce a clearer heatmap for the remaining features.

# In[98]:


multi_collinearity_heatmap(transformed_df.drop(list(transformed_df.columns[transformed_df.columns.str.startswith('city')]), axis=1), figsize=(25,22))


# Now it looks better without the attribute city. But we will keep both city and zipcode in our model.

# # Standardising and normalising¶

# In[99]:


numerical_columns = ['living_space','cold_rent','rent_per_square_meter', 'rooms', 'postcode']


# In[100]:


transformed_df[numerical_columns].hist(figsize=(10,11));


# In[101]:


# Separating X and y
X = transformed_df.drop('rent_per_square_meter', axis=1)
y = transformed_df.rent_per_square_meter

# Scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))


# In[102]:


# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# # 7. Model : Gradient boosted decision trees
#
# Xgboost is short for eXtreme Gradient Boosting package. Xgboost is a library which is designed and optimised for boosting tree algorithms. Extreme Gradient Boosting (xgboost) is same as gradient boosting framework but Xgboost is more efficient, flexible and portable. It is the package which is used to solve data science problems which include both linear model solver and tree learning algorithms.
#
# Gradient boosting is an approach where new models are created that predict the residuals or errors of prior models and then added together to make the final prediction. It is called gradient boosting because it uses a gradient descent algorithm to minimize the loss when adding new models.
# XGBoost (eXtreme Gradient Boosting) is an implementation of gradient boosted decision trees designed for speed and performance. Is a very popular algorithm that has recently been dominating applied machine learning for structured or tabular data.
#
# This model will most likely provide the best achievable accuracy and a measure of feature importance compared to our Hedonic regression (other than possible small accuracy increases from hyper-parameter tuning) due to XGBoost's superior performance.

# In[103]:


xgb_reg_start = time.time()

xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)
training_preds_xgb_reg = xgb_reg.predict(X_train)
val_preds_xgb_reg = xgb_reg.predict(X_test)

xgb_reg_end = time.time()

print(f"Time taken to run: {round((xgb_reg_end - xgb_reg_start)/60,1)} minutes")
print("\nTraining MSE:", round(mean_squared_error(y_train, training_preds_xgb_reg),4))
print("Validation MSE:", round(mean_squared_error(y_test, val_preds_xgb_reg),4))
print("\nTraining r2:", round(r2_score(y_train, training_preds_xgb_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_xgb_reg),4))


# This model looks good and our features explain approximately 99.88% of the variance in our target variable.

# In[104]:


y_test_array = np.array(list(y_test))
val_preds_xgb_reg_array = np.array(val_preds_xgb_reg)
hpm_df = pd.DataFrame({'Actual': y_test_array.flatten(), 'Predicted': val_preds_xgb_reg_array.flatten()})
hpm_df


# In[105]:


actual_values = y_test
plt.scatter(val_preds_xgb_reg, actual_values, alpha=.7,
            color='b') #alpha helps to show overlapping data
overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    (round(r2_score(y_test, val_preds_xgb_reg),4)),
                    (round(mean_squared_error(y_test, val_preds_xgb_reg))),4)
plt.annotate( s=overlay,xy=(5.5,2.5),size='x-large')
plt.xlabel('Predicted Price').set_color('blue')
plt.ylabel('Actual Price').set_color('red')
plt.title('SHP Regression Model')
plt.show()


# Our predicted values are almost identical to the actual values and fits perfectly. This graph would be the straight line y=x if each predicted value x would be equal to each actual value y.

# # 8. Feature importance
#
# Apart from its superior performance, a benefit of using ensembles of decision tree methods like gradient boosting is that they can automatically provide estimates of feature importance from a trained predictive model.
#
# Generally, importance provides a score that indicates how useful or valuable each feature was in the construction of the boosted decision trees within the model. The more an attribute is used to make key decisions with decision trees, the higher its relative importance.
#
# This importance is calculated explicitly for each attribute in the dataset, allowing attributes to be ranked and compared to each other.
#
# Importance is calculated for a single decision tree by the amount that each attribute split point improves the performance measure, weighted by the number of observations the node is responsible for. The performance measure may be the purity (Gini index) used to select the split points or another more specific error function.
# The feature importances are then averaged across all of the the decision trees within the model.

# In[106]:


ft_weights_xgb_reg = pd.DataFrame(xgb_reg.feature_importances_, columns=['weight'], index=X_train.columns)
ft_weights_xgb_reg.sort_values('weight', ascending=False, inplace=True)
ft_weights_xgb_reg.head(10)


# We can see the most important features from top to bottom.

# In[107]:


# Plotting feature importances
plt.figure(figsize=(10,25))
plt.barh(ft_weights_xgb_reg.index, ft_weights_xgb_reg.weight, align='center')
plt.title("Feature importances in the XGBoost model", fontsize=14)
plt.xlabel("Feature importance")
plt.margins(y=0.01)
plt.show()


# we can see the most important features in the plot above. We can also improve our model by removing the least important features and build the model for better accuracy and results.

# # 9. Ridge Regularization
#
# We can also try using Ridge Regularization to decrease the influence of less important features. Ridge Regularization is a process which shrinks the regression coefficients of less important features.
# We’ll once again instantiate the model. The Ridge Regularization model takes a parameter, alpha , which controls the strength of the regularization.
#
# We’ll experiment by looping through a few different values of alpha, and see how this changes our results.

# In[108]:


lr = linear_model.LinearRegression()

for i in range (-2, 3):
    alpha = 10**i
    rm = linear_model.Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, actual_values, alpha=.75, color='r')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                   round(ridge_model.score(X_test, y_test), 4),
                    round(mean_squared_error(y_train, training_preds_xgb_reg),4))
    plt.annotate( s=overlay,xy=(5.5,2.5),size='x-large')
    plt.show()


# These models performance doesn't improve our first model. In our case, adjusting the alpha did not substantially improve our model.

# # 10. XG Boost with dropped columns
#
# Let's try to improve the model by removing the city names

# In[109]:


# Read dataset
df = pd.read_csv('ckdnearest(df3, df2).csv')
df.head()


# In[110]:


# Dropping variables no longer needed
df.drop(['longitude', 'latitude', 'name', 'osm_id', 'Unnamed: 0', 'geometry', 'code', 'fclass', 'name', 'dist', 'city', 'publish_date', 'quarter' ], axis=1, inplace=True)


# In[111]:


df.head()


# In[112]:


transformed_df = pd.get_dummies(df)
transformed_df.head()


# In[113]:


multi_collinearity_heatmap(transformed_df, figsize=(20,20))


# In[114]:


numerical_columns = ['living_space','cold_rent','rent_per_square_meter', 'rooms', 'postcode']


# In[115]:


transformed_df[numerical_columns].hist(figsize=(10,11));


# In[116]:


# Separating X and y
X = transformed_df.drop('rent_per_square_meter', axis=1)
y = transformed_df.rent_per_square_meter

# Scaling
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=list(X.columns))


# In[117]:


# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)


# In[118]:


hpm_reg_start = time.time()

# Create instance of the model, `LinearRegression` function from
# Scikit-Learn and fit the model on the training data:

hpm_reg = LinearRegression()
hpm_reg.fit(X_train, y_train) #training the algorithm

# Now that the model has been fit we can make predictions by calling
# the predict command. We are making predictions on the testing set:
training_preds_hpm_reg = hpm_reg.predict(X_train)
val_preds_hpm_reg = hpm_reg.predict(X_test)

hpm_reg_end = time.time()

print(f"Time taken to run: {round((hpm_reg_end - hpm_reg_start)/60,1)} minutes")

# Check the predictions against the actual values by using the MSE and R-2 metrics:
print("\nTraining RMSE:", round(mean_squared_error(y_train, training_preds_hpm_reg),4))
print("Validation RMSE:", round(mean_squared_error(y_test, val_preds_hpm_reg),4))
print("\nTraining r2:", round(r2_score(y_train, training_preds_hpm_reg),4))
print("Validation r2:", round(r2_score(y_test, val_preds_hpm_reg),4))


# This model shows our features explain approximately 82.05% of the variance in our target variable which is less than the previous Xgboost model.

# In[119]:


y_test_array = np.array(list(y_test))
val_preds_hpm_reg_array = np.array(val_preds_hpm_reg)
hpm_df = pd.DataFrame({'Actual': y_test_array.flatten(), 'Predicted': val_preds_hpm_reg_array.flatten()})
hpm_df


# In[120]:


actual_values = y_test
plt.scatter(val_preds_hpm_reg, actual_values, alpha=.7,
            color='r') #alpha helps to show overlapping data
overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    (round(r2_score(y_test, val_preds_hpm_reg),4)),
                    (round(mean_squared_error(y_test, val_preds_hpm_reg))),4)
plt.annotate( s=overlay,xy=(5.5,2.5),size='x-large')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('SHP Regression Model')
plt.show()


# It doesn't look like removing the city attribute improved our model. It is almost identical as the Ridge Regularization.

# # Conclusion
#
# Xgboost model we created was the best amongst all, showed good results and predicted the rent per square meter accurately. We also tried to use Ridge Regularization but it doesn't improve the model. Ridge Regularization results are almost similar to the XG Boost with dropped columns results.
#
# We could also use other regression models and compare them. In our case Xgboost showed accurate results, the features explained approximately 99.88% of the variation in price with an RMSE of 73.4.
#
# The other shape files could also be downloaded and used for the better analysis. There are other shapefiles in the Geofabrik website such as railways, bus stops, buildings near by or lakes etc.
#
