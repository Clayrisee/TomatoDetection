import argparse
import os
import time
import urllib.request
from selenium import webdriver






def null_count(l):
    #given a list l, find the number of null
    null_count = 0
    
    for element in l:
        if element == None:
            null_count += 1
            
    return null_count