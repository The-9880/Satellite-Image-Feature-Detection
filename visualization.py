import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import ndimage
import cv2
import pickle

dict_roadtype = {
    "10":'Highway',
    "11":'Expressway',
    "12":'Primary Highway',
    "13":'Secondary Highway',
    "20":'Road',
    "21":'Arterial',
    "22":'Collector',
    "23":'Local',
    "24":'Alley/Lane/Utility',
    "25":'Connector/Ramp',
    "26":'Reserve/Trail',
    "27":'Rapid Transit',
    "28":'Planned',
    "29":'Strata',
    "80":'Bridge/Tunnel',
    "90":'Unknown',
    "95":'Unknown'
}


dict_roadtype_to_color = dict.fromkeys(["10","11","12","13"], 'green')
dict_roadtype_to_color.update(dict.fromkeys(["20","21","22","23","24","25"], 'red'))
dict_roadtype_to_color.update(dict.fromkeys(["26","27","28","29","80"], 'yellow'))
dict_roadtype_to_color.update(dict.fromkeys(["90","95"], 'magenta'))

FEATURES_KEY = 'features'
PROPERTIES_KEY = 'properties'
GEOMETRY_KEY = 'geometry'
COORDINATES_KEY = 'coordinates'
CLASS_KEY = 'CLASS'

MIN_NUM_PTS_PER_TILE = 4
PTS_PER_METER = 0.1

INPUT_FOLDER = 'data/basemap_tiles/'

filename_CLASS = 'data/shapefile/lrnf000r17a_e.json'
dict_features = json.load(open(filename_CLASS))[FEATURES_KEY]
d_tile_contents = defaultdict(list) # keys -> tiles. list of tile contents -> values
d_roadtype_tiles = defaultdict(set) # keys -> road types. tiles containing road types -> values


# Functions to use going forward -----
x_min = -79.40
y_min = 43.64
dx, dy = .02, .02
num_tiles_x, num_tiles_y = 100,100
total_no_tiles = num_tiles_x * num_tiles_y
x_max = x_min + num_tiles_x*dx
y_max = y_min + num_tiles_y*dy
BOUNDING_BOX = [x_min, y_min, x_max, y_max]

def coord_is_in_bb(coord, bb):
    x_mi = bb[0]
    y_mi = bb[1]
    x_ma = bb[2]
    y_ma = bb[3]
    print(coord)
    return coord[0] > x_mi and coord[0] < x_ma and coord[1] > y_mi and coord[1] < y_ma

def retrieve_roadtype(elem):
    return elem[PROPERTIES_KEY][CLASS_KEY]

def retrieve_coordinates(elem):
    buffer = elem[GEOMETRY_KEY][COORDINATES_KEY]# Our shapefile has a slightly different structure than the guide's
    buffer2 = []

    # I found a case that messed up the rest of the code
    # Because, for some reason, one of the coordinates was a list of even more coordinates; seems like a formatting issue
    # So this buffer step is to handle such anomalies.
    for it in buffer:
        if any(isinstance(el, list) for el in it):
            buffer2.extend(it)
            continue
        buffer2.append(it)

    return buffer2

def add_to_dict(d1, d2, coordinates, rtype):
    coordinate_ll_x = (coordinates[0] // dx) * dx # Rescale to make a proper multiple of dx
    coordinate_ll_y = (coordinates[1] // dy) * dy # Modified to match my coordinates, which are not ints.
    coordinate_ur_x = (coordinate_ll_x + dx)
    coordinate_ur_y = (coordinate_ll_y + dy)
    # Each tile in the dataset is named after its coordinates
    # The above transformations takes any coordinates and scales them
    # so that they become the coordinates of the tile they belong to
    # this allows us to map any coordinates to their corresponding tile in the dataset
    tile = "{}_{}_{}_{}.jpg".format(coordinate_ll_x, coordinate_ll_y, coordinate_ur_x, coordinate_ur_y)

    rel_coord_x = (coordinates[0] - coordinate_ll_x) / dx
    rel_coord_y = (coordinates[1] - coordinate_ll_y) / dy
    value = (rtype, rel_coord_x, rel_coord_y)
    d1[tile].append(value) # first dictionary maps tiles to their contents
    d2[rtype].append(tile) # second dictionary maps road types to unique sets of tiles

def calculate_intermediate_points(p1, p2, num_pts):
    dx = (p2[0]-p1[0]) / (num_pts + 1)
    dy = (p2[1]-p1[1]) / (num_pts + 1)
    # Each intermediate point from i=1...i=num_pts will be 1(i*dx, i*dy) distance away from the first point.
    return [[p1[0] + i * dx, p1[1] + i * dy] for i in range(1, num_pts+1)]

def euclidean_distance(p1, p2):
    diff = np.array(p1) - np.array(p2)
    return np.linalg.norm(diff)

def onehot_encode_labels(labels):
    labelsList = list(np.unique(labels)) # list of the unique road labels
    encoded = map(lambda x: labelsList.index(x), labels) # maps each road label to their class, given by their index in the unique list.
    return encoded

def randomize(dataset, labels1, labels2):
    permutation = np.random.permutation(dataset.shape[0])
    randomized_dataset = dataset[permutation, :, :, :] # randomizes along the axis of different examples
    randomized_labels1 = labels1[permutation] # Of course, all are shuffled the same way so that cases and labels aren't mismatched.
    randomized_labels2 = labels2[permutation]
    return randomized_dataset, randomized_labels1, randomized_labels2

def onehot_encode(np_array, num_unique_labels):
    return (np.arange(num_unique_labels) == np_array[:, None].astype(np.float32))

def reformat_data(dataset, labels1, labels2):
    dataset, labels1, labels2, = randomize(dataset, labels1, labels2)
    num_unique_labels1 = len(np.unique(labels1))
    num_unique_labels2 = len(np.unique(labels2))
    labels1 = onehot_encode(labels1, num_unique_labels1)
    labels2 = onehot_encode(labels2, num_unique_labels2)
    return dataset, labels1, labels2

# ------------------------------------

for elem in dict_features:
    print("Processing element...")
    coordinates = retrieve_coordinates(elem)
    rtype = retrieve_roadtype(elem)
    coordinates_in_bb = [coord for coord in coordinates if coord_is_in_bb(coord, BOUNDING_BOX)]
    if len(coordinates_in_bb) == 1:
        coord = coordinates_in_bb[0]
        add_to_dict(d_tile_contents, d_roadtype_tiles, coord, rtype)
    if len(coordinates_in_bb) > 1:
        add_to_dict(d_tile_contents, d_roadtype_tiles, coordinates_in_bb[0], rtype)
        for ii in range(1, len(coordinates_in_bb)):
            previous_coord = coordinates_in_bb[ii-1]
            coord = coordinates_in_bb[ii]
            add_to_dict(d_tile_contents, d_roadtype_tiles, coord, rtype)

            dist = euclidean_distance(previous_coord, coord)
            num_intermediate_points = int(dist/10)
            intermediate_coordinates = calculate_intermediate_points(previous_coord, coord, num_intermediate_points)
            for intermediate_coord in intermediate_coordinates:
                add_to_dict(d_tile_contents, d_roadtype_tiles, intermediate_coord, rtype)

print("Managing pyplot now...")
fig, axarr = plt.subplots(nrows=11, ncols=11, figsize=(16,16))
for ii in range(0,11):
    for jj in range(0,11):
        ll_x = x_min + ii*dx
        ll_y = y_min + jj*dy
        ur_x = ll_x + dx
        ur_y = ll_y + dy

        tile = "{}_{}_{}_{}.jpg".format(ll_x, ll_y, ur_x, ur_y)
        filename = INPUT_FOLDER + tile
        tile_contents = d_tile_contents[tile]

        ax = axarr[10-jj, ii]
        image = ndimage.imread(filename)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb_image)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        for elem in tile_contents:
            color = dict_roadtype_to_color[elem[0]] # tile_contents mapped the tilename to its (rtype, rel_coord_x, rel_coord_y) values. elem[0] retrieves rtype
            x = elem[1]*256 # scale the relative x coordinate up.
            y = (1-elem[2])*256 # scale the relative y coordinate up
            ax.scatter(x, y, c=color, s=10)

plt.subplots_adjust(wspace=0, hspace=0)
plt.show()

input("Continue on to train/val/test set creation")

import os

img_width = 256
img_height = 256
img_depth = 3
num_images = 10000

image_files = os.listdir(INPUT_FOLDER)

dataset = np.ndarray(shape=(num_images, img_width, img_height, img_depth), dtype=np.float32)
labels_roadtype = []
labels_roadpresence = np.ndarray(num_images, dtype=np.float32)

for counter, image in enumerate(image_files):
    filename = INPUT_FOLDER + image
    if image in list(d_tile_contents.keys()):
        tile_contents = d_tile_contents[image]
        roadtypes = sorted(list(set(elem[0] for elem in tile_contents)))
        print("ROADTYPES: ", roadtypes)
        roadtype = "_".join(roadtypes)
        labels_roadpresence[counter] = 1 # road is present in the tile.
    else:
        print("NOT FOUND OR NEGATIVE:", image)
        roadtype = ''
        labels_roadpresence[counter] = 0
    labels_roadtype.append(roadtype)

unique, counts = np.unique(labels_roadpresence, return_counts=True)
print(dict(zip(unique, counts))) # Quick check on the ratio of positive-to-negative cases for road detection
                                 # for road extraction/tracing, however, this shouldn't really matter.

onehot_labels_roadtype = np.array(list(onehot_encode_labels(labels_roadtype)))
dataset, labels_roadpresence, onehot_labels_roadtype = reformat_data(dataset, labels_roadpresence, onehot_labels_roadtype)



# Step 4.2 - Saving datasets as pickle

train_start = 0
val_start = 1200
test_start = 1600
total_images = 2000

output_pickle_file = 'data/satellite_data.pickle'

with open(output_pickle_file, 'wb') as f:
    save = {
        'train_dataset': dataset[train_start:val_start, :, :, :],
        'train_labels_roadtype': labels_roadtype[train_start:val_start],
        'train_labels_roadpresence': labels_roadpresence[train_start:val_start],
        'valid_dataset': dataset[val_start:test_start, :, :, :],
        'valid_labels_roadtype': labels_roadtype[val_start:train_start],
        'valid_labels_roadpresence': labels_roadpresence[val_start:test_start],
        'test_dataset': dataset[test_start:total_images, :, :, :],
        'test_labels_roadtype': labels_roadtype[test_start:total_images],
        'test_labels_roadpresence': labels_roadpresence[test_start:total_images]
    }
    pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)

print("\nDataset saved to {}".format(output_pickle_file))
print("Num entries in d_tile_contents:", len(d_tile_contents.keys()))