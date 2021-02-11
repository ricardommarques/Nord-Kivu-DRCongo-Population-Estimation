
import pandas as pd
import arcpy
import matplotlib.pyplot as plt
import numpy as np

# Import data
##################################
print('\nImporting data...')
workspace = "C:\Users\\Population_Estimation\\arcpro_Whole_Country\MyProject.gdb"
pop_data=workspace+'\\popDensity_NordKivu'
core_data=workspace+'\Cores_NordKivu'
csv="C:\Users\\Population_Estimation\Python"

print('\nExtracting popDensity data...')
arcpy.conversion.TableToTable(pop_data, csv, 'Pop_NordKivu.csv')
print('\nExtracting cores data...')
arcpy.conversion.TableToTable(core_data, csv, 'Core_NordKivu.csv')

print('\nFinished Successfully!')