#!/usr/bin/env python
# coding: utf-8

## run in python >= 3.8


# Imports
import os
import pickle
import math
import shutil
import uuid
import numpy as np
import pandas as pd
import plotnine as p9


######################################################################
###
### Load and import PeakBot
##
#

## Import PeakBot from directory not as installed package (can be omitted if the package is installed)
import sys
sys.path.append(os.path.join("..", "peakbot", "src"))

## Load the PeakBot package
import peakbot.Chromatogram
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog
import peakbot


## Function for loading mzXML files (and saving them as pickle file for fast access)
def loadFile(path):
    tic()
    mzxml = None
    if os.path.exists(path+".pickle"):
        with open(path+".pickle", "rb") as inF:
            mzxml = pickle.load(inF)
        print("Imported chromatogram.pickle for '%s'"%(path))
    else:
        mzxml = peakbot.Chromatogram.Chromatogram()
        mzxml.parse_file(path)
        with open(path+".pickle", "wb") as outF:
            pickle.dump(mzxml, outF)
        print("Imported chromatogram for '%s'"%(path))
    return mzxml


###############################################
### Process files
##
if __name__ == "__main__":
    tic(label="overall")

    ###############################################
    ### chromatograms to process
    ##
    ## Different LC-HRMS chromatograms can be used for generating a training or validation dataset
    ##
    ## file: Path of the mzXML file
    ## params: parameter collection for the particular sample (see variable expParams)
    files = {
        "670_Sequence3_LVL1_1"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_1.mzXML", "polarity": "Q Exactive (MS lvl: 1, pol: +)"},
        "670_Sequence3_LVL1_2"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_2.mzXML", "polarity": "Q Exactive (MS lvl: 1, pol: +)"},
        "670_Sequence3_LVL1_3"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_3.mzXML", "polarity": "Q Exactive (MS lvl: 1, pol: +)"},
        "08_EB3391_AOH_p_60"    : {"file": "./Data/PHM/08_EB3391_AOH_p_60.mzXML", "polarity": "LTQ Orbitrap Velos (MS lvl: 1, pol: +)"}
    }

    print("")
    for fi in files.keys():
        print("Processing file '%s', polarity '%s'"%(fi, files[fi]["polarity"]))
        
        mzxml = loadFile(files[fi]["file"])
        mzxml.keepOnlyFilterLine(files[fi]["polarity"])
        mzxml.removeNoise(1E5)
                
        val = peakbot.estimateParameters(mzxml, files[fi]["file"].replace(".mzXML", ""))
        
        print("Suggested parameters: intraScanMaxAdjacentSignalDifferencePPM = %.3f, interScanMaxSimilarSignalDifferencePPM = %.3f"%(val, val))
        print("")
