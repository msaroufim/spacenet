from PIL import Image
import numpy as np
import pickle
import os

# Source directories for data and label
DATA_DIR  = 'tiles/'
LABEL_DIR = 'masks/'
NUM_EXAMPLES = 100

# Data structures - X data, Y labels
X = []
Y = []

# Read each element from each directory and append them as arrays to a list
def setup_datastructure_from_tif(directory,data_structure,i=0):
  for tile_name in os.listdir(directory):
    tile_tif = Image.open(directory + tile_name)
    data_structure.append(np.array(tile_tif))
    i += 1
    if i > NUM_EXAMPLES:
      break

setup_datastructure_from_tif(DATA_DIR,X)
setup_datastructure_from_tif(LABEL_DIR,Y)

# Save data structures to disk

def pickle_it(dataset,destination_name='dataset.pickle'):
      with open(destination_name, 'wb') as f:
                pickle.dump(dataset, f)   


# since mask is blue can throw out all the other channels
# since individual value is either 0 or 255 can replace this by 0 or 1
targets = [Y[i][:,:,2] / 255  for i in range(len(Y))]

pickle_it(X,'tiles.pickle')
pickle_it(targets,'targets.pickle')

