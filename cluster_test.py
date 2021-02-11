
import pandas as pd
import arcpy
import matplotlib.pyplot as plt
import numpy as np
import psutil

#Create DataFrames
#####################

print('\nCreating dataframes...')

pop_csv="C:\Users\\Population_Estimation\Python\Pop_NordKivu.csv"
core_csv="C:\Users\\Population_Estimation\Python\Core_NordKivu.csv"

pop= pd.DataFrame(pd.read_csv(pop_csv))[['Lat', 'Lon', 'Population']]
core=pd.DataFrame(pd.read_csv(core_csv))[['osm_id', 'name','Lat', 'Lon']]
core=core[core['name']!=' ']
core=core.reset_index()

#plt.plot(pop['Lon'],pop['Lat'],'.',markersize=1)
#plt.plot(core['Lon'],core['Lat'],'or',markersize=5)
#plt.show()

#Perform clusters
######################
print('\nPreparing pop data for DBSCAN...')

peso=12
max_dist=0.0009

X=[] #list to be used in DBSCAN
for ele in range(len(pop)):
    X.append([pop['Lon'][ele],pop['Lat'][ele]])


print('\nCreating sample weights list...')
weights=[] #add weight for forced cluster cores
core_names=[] #add osm names for forced cluster cores
core_ids=[] #add osm id for forced cluster cores

for sample in X:
    if sample[0] in core.Lon.tolist() and sample[1] == core.Lat.tolist()[core.Lon.tolist().index(sample[0])]:
        weights.append(peso)
        core_names.append(core['name'][core.Lon.tolist().index(sample[0])])
        core_ids.append(core['osm_id'][core.Lon.tolist().index(sample[0])])
    else:
        weights.append(1)
        core_names.append('---')
        core_ids.append(0)

#Performing Density Based Clustering
print('\nPerforming Density Based Clustering...')
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=max_dist, min_samples=peso).fit(X,sample_weight=weights)
labels = db.labels_ #Assign label columns to points

# Create OSM places table
###############
print('\nCreating OSM places table...')
unique_labels = set(labels)
cluster_label=[]
cluster_pop=[]
cluster_osm=[]
cluster_name=[]

print('\nAdding population values...')
for k in unique_labels:

    class_member_mask = (labels == k)
    for i in list(np.where(labels==k)[0]):
       if weights[i]==peso:
           cluster_label.append(k)
           cluster_pop.append(sum(pop['Population'][class_member_mask]))
           cluster_osm.append(str(core_ids[i]))
           cluster_name.append(core_names[i])


#Taking lists values into dataframe and erasing lists
print('\nCreating OSM places dataframe / erasing lists...')

OSM_places= pd.DataFrame({'OSM_ID': cluster_osm})
OSM_places['Name']= cluster_name
OSM_places['Label']= cluster_label
OSM_places['Total Population']=cluster_pop

OSM_places.to_csv('OSM_places.csv', index=False)
print('\nOSM_places saved in csv file')


# Plot result
######################
print('\nPlotting results from clustering analysis...')
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1: # Black used for noise.
        class_member_mask = (labels == k)
        xy = np.array(X)[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], '.', markerfacecolor=tuple([0, 0, 0, 1]), markersize=1)
    else:
       class_member_mask = (labels == k)
       xy = np.array(X)[class_member_mask]
       plt.plot(xy[:, 0], xy[:, 1], '.', marker='.',   markerfacecolor=tuple(col), markersize=3)

       centroid_x = sum(xy[:, 0]) / len(xy)
       centroid_y = sum(xy[:, 1]) / len(xy)
       #plt.plot(centroid_x, centroid_y, 'oy',markersize=2)

plt.plot(core['Lon'],core['Lat'],'or',markersize=5)
plt.show()

print('\nFinished Successfully!')