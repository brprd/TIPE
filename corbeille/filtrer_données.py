#crée un fichier avec seulement les lignes répondant aux critères suivants

input_file='données.csv'
output_file='ais_data.csv'
heures=('00','01','02','03','04') #heures à sélectionner
#carré dans lequel les données vont être conservées
latitude_min=56
latitude_max=64
longitude_min=-154
longitude_max=-146


import csv

with open(input_file, 'r', newline='') as input:
    with open(output_file, 'w', newline='') as output:
        input.readline() #la premiere ligne ne contient pas les données
        csvreader = csv.reader(input, delimiter=',')
        csvwriter = csv.writer(output, delimiter=',')
        for row in csvreader:
            mmsi=row[0]
            hour=row[1][11:13]
            lat=float(row[2])
            lon=float(row[3])
            if (hour in heures) and lat>=latitude_min and lat<=latitude_max and lon>=longitude_min and lon<=longitude_max: #vérification de tous les critères
                csvwriter.writerow(row)

