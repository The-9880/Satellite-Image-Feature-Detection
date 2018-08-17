from owslib.wms import WebMapService
URL = "http://maps.geogratis.gc.ca/wms/canvec_en?request=getcapabilities&service=wms&version=1.3.0"
mapService = WebMapService(URL, version="1.1.1")

OUTPUT_DIR = 'data/image_tiles/'

x_min = -79.409566
y_min = 43.642541
dx, dy = .02, .02
num_tiles_x, num_tiles_y = 100,100
total_no_tiles = num_tiles_x * num_tiles_y

x_max = x_min + num_tiles_x*dx
y_max = y_min + num_tiles_y*dy
BOUNDING_BOX = [x_min, y_min, x_max, y_max]

# Setup has 100 tiles along the x and y directions, each tile being 200x200 -- therefore the bounding box is 20,000 units along both axes

# Flags to control data collection while I'm messing around
targetDataCollected = True
inputDataCollected = False

# Layer: road_segment_50k. ESPG: 4269 -- This one seems more specific to roads for cars and regular motor vehicles
if not targetDataCollected:
    for ii in range(0, num_tiles_x):
        print('Target Data ', ii+1)
        for jj in range(0, num_tiles_y):
            # ll meaning lower-left
            ll_x_ = x_min + ii*dx
            ll_y_ = y_min + jj*dy
            bbox = (ll_x_, ll_y_, ll_x_ + dx, ll_y_ + dy)
            img = mapService.getmap(layers=['road_segment_50k'], srs='EPSG:4269', bbox = bbox, size=(256,256), format='image/jpeg', transparent=True)
            filename = "{}_{}_{}_{}.jpg".format(bbox[0], bbox[1], bbox[2], bbox[3])
            with open(OUTPUT_DIR + filename, 'wb') as f:
                f.write(img.read())

OUTPUT_DIR = 'data/basemap_tiles/'
URL = "http://geoappext.nrcan.gc.ca/arcgis/services/BaseMaps/CBMT3978/MapServer/WMSServer?request=GetCapabilities&service=WMS"
mapService = WebMapService(URL, version="1.1.1")

if not inputDataCollected:
    for ii in range(0, num_tiles_x):
        print('Input Data ', ii+1)
        for jj in range(0, num_tiles_y):
            # ll meaning lower-left
            ll_x_ = x_min + ii * dx
            ll_y_ = y_min + jj * dy
            bbox = (ll_x_, ll_y_, ll_x_ + dx, ll_y_ + dy)
            img = mapService.getmap(layers=['1'], srs='EPSG:4326', bbox=bbox, size=(256, 256), format='image/jpeg',
                                    transparent=True)
            filename = "{}_{}_{}_{}.jpg".format(bbox[0], bbox[1], bbox[2], bbox[3])
            with open(OUTPUT_DIR + filename, 'wb') as f:
                f.write(img.read())