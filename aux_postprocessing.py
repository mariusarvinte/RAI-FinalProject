# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:23:29 2019

@author: ma56473
"""

import re
# Read multi-try output file and return accuracies in 1-D vector
# NOTE: This only works when there is only a single x.y floating number per line
def parse_tries(filename):
    accuracy = []
    for line in open(filename):
        matches = re.findall("[+-]?\d+\.\d+", line)
        accuracy.append(float(matches[0]))
    return accuracy