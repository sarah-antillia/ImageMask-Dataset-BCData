# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# 2024/07/03
# mageMaskDatasetGenerator:
import os
import numpy as np
import cv2
# pip install h5py
import h5py
import traceback

class H5AnnotationParser:

  def __init__(self, radius=20, output_dir="./test"):
    self.W = 640
    self.H = 640
    self.circle_radius=radius
    self.output_dir   = output_dir


  def parse(self, h5_file):
     print("=== h5_annotation {}".format(h5_file))
     print("=== Poitwise Annotation ")
     print("--- Draw a circle of radius {} at each center in the pointwise annotation ".format(self.circle_radius))
     if os.path.exists(h5_file):
       # 1 Black background 
       black_background = np.zeros((self.W, self.H, 1))
       mask = black_background
       keys = []
       with h5py.File(h5_file, "r") as f:
         f.visit(keys.append) # append all keys to list
         print("=== keys {}".format(keys))
         key = 'coordinates'
         if len(keys) == 1:
           key = keys[0]
         print("--- key {}".format(key))
         coordinates = np.asarray(f[key])
         print("=== Pointwise annotation")
         for point in coordinates:
           [x, y] = point
           print(point)
           cv2.circle(img=mask, center =(x, y), radius =self.circle_radius, color =255, thickness=-1)

       basename = os.path.basename(h5_file)
       basename = basename.replace(".h5", ".jpg")
       mask_filepath = os.path.join(self.output_dir,  basename)
       cv2.imwrite(mask_filepath, mask)
       print("=== Saved {}".format(mask_filepath))
  
if __name__ == "__main__":
  try:
    radius     = 20
    output_dir = "./output"
    h5_file    = "./test/annotations/0.h5"

    parser = H5AnnotationParser(radius=radius, output_dir=output_dir)
    parser.parse(h5_file)

  except:
    traceback.print_exc()
