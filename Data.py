import json

with open('events.json', 'r') as f:
    for row in f:
        data = row
        data.replace("\t", "")
        data = data.split(':')
        print(data)