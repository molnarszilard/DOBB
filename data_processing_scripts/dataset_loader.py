# This file is part of 3D-Aware-Ellipses-for-Visual-Localization.
#
# Author: Matthieu Zins (matthieu.zins@inria.fr)
#
# 3D-Aware-Ellipses-for-Visual-Localization is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3
# of the License, or (at your option) any later version.
#
# 3D-Aware-Ellipses-for-Visual-Localization is distributed in the hope that
# it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with 3D-Aware-Ellipses-for-Visual-Localization.
# If not, see <http://www.gnu.org/licenses/>.

import json
import numpy as np


class Dataset_loader:
    """
        Interace with our own dataset file format.
    """

    def __init__(self, filename):
        with open(filename, "r") as fin:
            self.dataset = json.load(fin)
        self.nb_images = len(self.dataset)

    def __len__(self):
        return len(self.dataset)

    def get_image_size(self, idx):
        """
            Get the image size.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None, None
        return self.dataset[idx]["width"], self.dataset[idx]["height"]
    
    def get_K(self, idx):
        """
            Get the K matrix.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        return np.asarray(self.dataset[idx]["K"])


    def get_Rt(self, idx):
        """
            Get the extrinsic parameters.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        R = np.asarray(self.dataset[idx]["R"])
        t = np.asarray(self.dataset[idx]["t"])
        return np.hstack((R, t.reshape((3, 1))))


    def get_rgb_filename(self, idx):
        """
            Get the rgb image filename.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        return self.dataset[idx]["file_name"]

    def get_annotations(self, idx):
        """
            Get objects annotations.
        """
        if idx < 0 or idx >= self.nb_images:
            print("Invalid index")
            return None
        if "annotations" not in self.dataset[idx].keys():
            print("No annotations available")
            return None
        return self.dataset[idx]["annotations"]