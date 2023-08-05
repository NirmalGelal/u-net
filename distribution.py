import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob

image_dir = "data/training_data/masks"
data_path = os.path.join(image_dir, '*g')
files = glob.glob(data_path)

yellow_count = 0
magenta_count = 0
cyan_count = 0
black_count = 0
blue_count = 0
white_count = 0
green_count = 0

# count = 0

for file in files:
    # count+=1

    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cyan_count += np.count_nonzero((img == [0, 255, 255]).all(axis=2))      #urban_land
    yellow_count += np.count_nonzero((img == [255,255, 0]).all(axis=2))     #agriculture
    magenta_count += np.count_nonzero((img == [255, 0,255]).all(axis=2))    #range_land
    green_count += np.count_nonzero((img == [0, 255,0]).all(axis=2))        #forest_land
    blue_count += np.count_nonzero((img == [0, 0, 255]).all(axis=2))        #water
    white_count += np.count_nonzero((img == [255, 255, 255]).all(axis=2))   #barren_land
    black_count += np.count_nonzero((img == [0, 0, 0]).all(axis=2))         #unknown

    # if(count>10):
    #     break

print("yellow:",yellow_count/1e6)
print("magenta:",magenta_count/1e6)
print("cyan:",cyan_count/1e6)
print("black:",black_count/1e6)
print("white:",white_count/1e6)
print("blue:",blue_count/1e6)
print("green:",green_count/1e6)

df = pd.DataFrame({
    'Land Type': ['Urban Land', 
                  'Agriculture', 
                  'Rangeland', 
                  'Forest Land', 
                  'Water', 
                  'Barren Land', 
                  'Unknown'],
    'Pixels': [cyan_count, 
               yellow_count, 
               magenta_count, 
               green_count, 
               blue_count, 
               white_count, 
               black_count]
})

ax = df.plot.bar(x='Land Type', y='Pixels', width=1)
plt.show()