# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 09:45:50 2021

@author: thiba
"""

import os
import json

list_dico = {}

for file in os.listdir('.'):
    if 'mmp' in file:
        print(file)
        with open(file, 'r') as f:
            data = json.load(f)
            
        list_dico.update(data)

with open('mmp_discrete_sanitized.json', 'w+') as fp:
    json.dump(list_dico, fp, indent=2)
