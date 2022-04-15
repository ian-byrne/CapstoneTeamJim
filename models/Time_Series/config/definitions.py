# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 21:23:32 2022

@author: melan

This short piece of code will always calculate the correct path to the root directory of our project.

"""
import os

def root_directory():
    return os.path.realpath(os.path.join(os.path.dirname('__file__'), '..'))

# if __name__ == '__main__':

    
#     root_directory()
    # ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname('__file__'), '..'))