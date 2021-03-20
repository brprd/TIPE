#crée un fichier avec seulement les lignes répondant aux critères suivants

input_file='../ais_data.csv'
output_file='../bdd2.csv'
heures=('00','01','02','03','04') #heures à sélectionner
#carré dans lequel les données vont être conservées
latitude_min = -90
latitude_max = 90
longitude_min = -180
longitude_max = 180


import csv

with open(input_file, 'r', newline='') as input:
    with open(output_file, 'w', newline='') as output:
        input.readline() #la premiere ligne ne contient pas les données
        csvreader = csv.reader(input, delimiter=',')
        csvwriter = csv.writer(output, delimiter=',')
        for row in csvreader:
            mmsi=row[0]
            hour=row[1][11:13]
            time=row[1]
            lat=float(row[2])
            lon=float(row[3])
            SOG=float(row[4])
            COG=float(row[5])

            if (hour in heures) and lat>=latitude_min and lat<=latitude_max and lon>=longitude_min and lon<=longitude_max: #vérification de tous les critères
                csvwriter.writerow([time,mmsi,lat,lon,SOG,COG])

