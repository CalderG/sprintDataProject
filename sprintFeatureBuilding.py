import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# cleanerSprintFrame.csv
# altitudes4Venues.csv
editingSprintFrame = pd.read_csv("C:\Users\calde\OneDrive\Documents\sprintData\cleanerSprintFrame.csv", low_memory = False)

altitudes2Add = pd.read_csv("C:/Users/calde/OneDrive/Documents/sprintData/altitudes4Venues.csv")

# Time Correction Function - credited from Jonas Mureika's paper: 
# "Back of the envelope" wind and altitude correction for 100 metre sprint times, 22.6.2000

def time_correction(time, wind, altitude):
    return time * (1.027 - 0.027 * np.exp(-0.000125 * altitude) * (1 - ((wind * time) / 100))**2)

# Convert the times to float

# Pseudocode:

# Check result column, if it contains h then remove the h, convert to float and add 0.24
# otherwise if it contains a or A, remove the a or A, and convert to float

# Removing altitude time indicators and + sign for times on route to 150m or 200m
editingSprintFrame["result"] = editingSprintFrame["result"].replace(to_replace = "a$",value ="",regex=True)
editingSprintFrame["result"] = editingSprintFrame["result"].replace(to_replace = "A$",value ="",regex=True)
editingSprintFrame["result"] = editingSprintFrame["result"].replace(to_replace = "\\+$",value ="",regex=True)

# Make an indicator where it's a 1 if there's a handtime, then for rows with a 1, remove the h

editingSprintFrame["removeHandTime"] = editingSprintFrame["result"].str.contains("h")
editingSprintFrame.loc[(editingSprintFrame["removeHandTime"] == True), "result"] = editingSprintFrame["result"].replace(to_replace = "h$", value = "", regex = True)


# Modifying rows with issues in the results as the times are completely messed up

editingSprintFrame.loc[12080, "result"] = "10.20"
editingSprintFrame.loc[13803, "result"] = "10.61"
editingSprintFrame.loc[62763, "result"] = "10.46"

# Convert results to float and add 0.24 for the hand times

editingSprintFrame["result"] = pd.to_numeric(editingSprintFrame["result"], downcast = "float")
editingSprintFrame.loc[(editingSprintFrame["removeHandTime"] == True), "result"] = editingSprintFrame["result"] + 0.24


# Rename the columns in order to make a nice dictionary
altitudes2Add = altitudes2Add.rename(columns={"Unnamed: 0": "venue", "0": "elevation"})

altitudes2Fix = dict(zip(altitudes2Add["venue"], altitudes2Add["elevation"]))

# Create a new column which is the result of mapping the series to the dictionary keys
editingSprintFrame["altitude"] = editingSprintFrame["venue"].map(altitudes2Fix)

# Fix outlier at index 79104 - data entry error, correct wind reading on World Athletics -> -1.3 m/s

editingSprintFrame.loc[79104, "wind"] = -1.3

# Create the time corrected results column by passing in the result column for time, wind for wind, and altitude
# for altitude
editingSprintFrame["time-corrected"] = time_correction(editingSprintFrame["result"], editingSprintFrame["wind"], editingSprintFrame["altitude"])

