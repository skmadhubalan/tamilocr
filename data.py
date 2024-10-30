import os
import shutil
from PIL import Image, ImageFilter, ImageOps
import numpy as np
from scipy import ndimage

def process(img):
    conv = img.convert("L")
    inv = ImageOps.invert(conv)
    pad = ImageOps.expand(inv, 2)
    thick = pad.filter(ImageFilter.MaxFilter(5))
    ratio = 40 / max(thick.size)
    new_size = tuple([int(round(x*ratio)) for x in thick.size])
    res = thick.resize(new_size, Image.LANCZOS)

    arr = np.asarray(res)
    com = ndimage.measurements.center_of_mass(arr)
    result = Image.new("L", (64, 64))
    box = (int(round(32 - com[1])), int(round(32 - com[0])))
    result.paste(res, box)
    return result

train_root = 'data/unprocessed/train/'
test_root = 'data/unprocessed/test/'
train_target = 'data/processed/train/'
test_target = 'data/processed/test/'

for i in range(156):
    os.makedirs(os.path.join(train_target, str(i)), exist_ok=True)
    os.makedirs(os.path.join(test_target, str(i)), exist_ok=True)

train_count = 0
for root, dirs, files in os.walk(train_root):
    for dir_name in dirs:
        for dir_root, subdirs, dir_files in os.walk(os.path.join(root, dir_name)):
            for file_name in dir_files:
                if file_name == "Thumbs.db":
                    continue
                file_path = os.path.join(dir_root, file_name)
                label = file_name[:3]
                new_name = f"{label}u{dir_name[4:]}{file_name[3:]}"
                location = os.path.join(train_target, str(int(label)), new_name)
                shutil.copy(file_path, location)
                train_count += 1
print(f"{train_count} training examples")

test_count = 0
with open('data/ground_truth.txt', 'r') as f:
    for i, line in enumerate(f):
        label = line[6:-1]
        new_name = f"{str(i).zfill(5)}.tiff"
        location = os.path.join(test_target, label, new_name)
        shutil.copy(os.path.join(test_root, f"{i:05}.tiff"), location)
        test_count += 1
print(f"{test_count} test examples")
