# -*- coding: utf-8 -*-

# To be used for getting image paths for the below format:

# └── dataset
#     ├── training_set
#     │   ├── cats
#     │   └── dogs
#     └── test_set
#         ├── cats
#         └── dogs

import os

# Valid image extensions
im_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')

def gen_impaths(path, valid_exts=None, train_justify=False):    
    for i, (_root, _dirs, _files) in enumerate(os.walk(path)):
        # Check to read training set data first
        if i == 0 and train_justify is True:
            for _dir in _dirs:
                if 'train' in _dir.lower():
                    _dirs.insert(0, _dirs.pop(_dirs.index(_dir)))
        
        for _file in _files:
            # Get the file extension
            ext = _file[_file.rfind('.'):].lower()
            
            # Check for valid image extensions
            if valid_exts is None or ext.endswith(valid_exts):
                impath = os.path.join(_root, _file)
                yield impath

def get_impaths(path, train_justify=False):
    # only return images with valid extensions
    return list(gen_impaths(path, valid_exts=im_exts, train_justify=train_justify))
                
            
            


