import pandas as pd
import matplotlib.pyplot as plt
import requests

rawSprintFrame = pd.read_csv("sprintTestFrame.csv", low_memory = False)

# Check dimensions and emptiness of columns

rawSprintFrame.shape
# 116328 rows, 23 cols

for each_col in range(len(rawSprintFrame.columns)):
    print(rawSprintFrame.columns[each_col] + ": " + f'{rawSprintFrame.iloc[0:116327, each_col].count()}')

# Remove the following columns: field1, rank, secondaryCountry, 
# nameLong, and ranking for lack of data

# Records may be kept or removed later

# Axis set to 1 to specify columns
cleanerSprintFrame = rawSprintFrame.drop(['field1', 'rank', 'secondaryCountry', 
'nameLong', 'ranking'], axis = 1)

# Remove rows with no wind readings
cleanerSprintFrame = cleanerSprintFrame.dropna(subset = "wind")

# Remove rows with no known venues

cleanerSprintFrame = cleanerSprintFrame[cleanerSprintFrame["venue"] != "UNKNOWN"]

# Build dictionary of venues to find altitudes for

venuesList = list(cleanerSprintFrame.loc[0:110757, "venue"].unique())

altitudes = []

## Current Altitude Game Plan: Get lat and long then input into other api to get elevation

base_Lat_Long = "https://geocode.maps.co/search?q="

# https://tessadem.com/api/elevation?key=
base_Elevation = "https://www.elevation-api.eu/v1/elevation/"

# build list of blank altitudes
for every_venue in venuesList:
    newUrl = base_Lat_Long + every_venue + "&api_key=6a114cc3b9249064075425gmza4e6eb"
    latLong_Response = requests.get(newUrl)
    try:
        potentialResponse = latLong_Response.json()
        givenLat = potentialResponse[0]["lat"]
        givenLong = potentialResponse[0]["lon"]
        finalUrl = base_Elevation + givenLat + "/" + givenLong
        altitudeExtraction = requests.get(finalUrl)
        altitudes.append(altitudeExtraction.json())
    except:
        print(f"No altitude found for {every_venue}")
        altitudes.append("Replace this with manually found altitude")
   

# Finding the indices where no altitude was found - quick line from Stack Overflow

venueindices = [i for i, x in enumerate(altitudes) if x == "Replace this with manually found altitude"]

# Missing altitudes (found manually through google)
missingAltitudes = [715.0,57.0,42.0,93.0,76.0,30.0,63.0,2.0,263.0,117.0,133.0,1543.0,502.0,83.0,18.0,
                    959.0,4.0,21.0,12.0,19.0,61.0,27.172,7.0,79.0,157.0,1160.0,37.0,490.0,19.0,48.0,
                    467.0,2.0,157.0,17.0,107.0,1593.0,111.0,30.0]

# Filling all the altitudes in
starting_altitude = 0
for each_indice in venueindices:
    altitudes[each_indice] = missingAltitudes[starting_altitude]

# Make a dictionary and save the cleaner dataframe + dictionary

venueAltitudePairs = dict(zip(venuesList, altitudes))

cleanerSprintFrame.to_csv("cleanerSprintFrame.csv")

venueAltitudeFrame = pd.DataFrame.from_dict([venueAltitudePairs])

testVenueFrame = venueAltitudeFrame.T

testVenueFrame.to_csv("altitudes4Venues.csv")