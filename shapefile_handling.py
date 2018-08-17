import shapefile
import json

from datetime import date
from datetime import datetime

def JSONencoder(obj):
    if isinstance(obj, (datetime, date)):
        serial = obj.isoformat()
        return serial
    if isinstance(obj, bytes):
        return {'__class__': 'bytes', '__value__':list(obj)}
    raise TypeError("Type %s is not serializable" % type(obj))

input_filename = 'data/shapefile/lrnf000r17a_e.shp'
output_filename = 'data/shapefile/lrnf000r17a_e.json'

reader = shapefile.Reader(input_filename)
fields = reader.fields[1:]
# print(fields)
field_names = [field[0] for field in fields]
buffer = []

shapeRecords = reader.iterShapeRecords()
for sr in shapeRecords:
    print("CASE")
    atr = dict(zip(field_names, sr.record))
    geom = sr.shape.__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr))

with open(output_filename, 'w') as f:
    f.write(json.dumps({"type":"FeatureCollection", "features":buffer}, indent=2, default=JSONencoder) + "\n")