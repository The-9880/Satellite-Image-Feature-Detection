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

# First, I ran into MemoryError constantly. After trying iterShapeRecords and still failing, I realized I was using a 32-bit interpreter
# switched to a 64-bit interpreter, and watched as >8 GB of RAM was used up before a system crash
# So now I'm just going to save this stuff incrementally, in a manually-programmed way.

counter = 0
for sr in reader.iterShapeRecords():
    if len(buffer) == 100000:
        with open(output_filename, 'a') as f:
            if not counter:
                f.write(json.dumps({"type": "FeatureCollection", "features": buffer}, default=JSONencoder)[:-2])
                counter += 1
            else:
                f.write(', ' + json.dumps(buffer, default=JSONencoder)[1:-1])
        buffer = []
        print(counter)
        counter += 1

    atr = dict(zip(field_names, sr.record))
    geom = sr.shape.__geo_interface__
    buffer.append(dict(type="Feature", geometry=geom, properties=atr))

if len(buffer):
    with open(output_filename, 'a') as f:
        f.write(', ' + json.dumps(buffer, default=JSONencoder)[1:] + '}')
    buffer=[]
else:
    f.write(']}')


# with open(output_filename, 'w') as f:
#     f.write(json.dumps({"type":"FeatureCollection", "features":buffer}, indent=2, default=JSONencoder) + "\n")
