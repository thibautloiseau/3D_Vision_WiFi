# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 16:59:48 2021

@author: thiba
"""

import os
import numpy as np
import json
import process_csi as process
import time

dico_mmp = {}

for idx, doc in enumerate(os.listdir("acquisitions/03-03-2021_grosse_chambre_Thibaut")):
    if doc in ['25', '30', '35', '40', '45', '5', '50']:
        print(doc)
        start = time.time()
        CSI = process.CSI("acquisitions/03-03-2021_grosse_chambre_Thibaut/" + doc)
        no_acq = CSI.get_raw_data().shape[0]
        
        dico_mmp[doc] = {}
        
        thetas_1 = np.zeros(shape=no_acq)
        thetas_2 = np.zeros(shape=no_acq)
        
        for paquet in range(no_acq):
            print('\t' + str(paquet))
            thetas = CSI.DoA_MMP(paquet, 2.7e-2)[0]
            thetas_1[paquet] = thetas[0]
            thetas_2[paquet] = thetas[1]
        
        dico_mmp[doc]['1'] = thetas_1.tolist()
        dico_mmp[doc]['2'] = thetas_2.tolist()
        
        with open('mmp_discrete_sanitized_4.json', 'w+') as fp:
            json.dump(dico_mmp, fp, indent=2)
                
        print(time.time()-start)