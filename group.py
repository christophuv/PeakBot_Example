#!/usr/bin/env python
# coding: utf-8

## run in python >= 3.8
## activate conda environment on jucuda
## conda activate python3.8

# Imports
import os
import pickle
import tempfile
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.client import device_lib 
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)




######################################################################
###
### Load and import PeakBot
##
#

import sys
sys.path.append(os.path.join("..", "peakbot", "src"))
import peakbot
import peakbot.Chromatogram
import peakbot.cuda
            
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog







###############################################
### Process files 
##
if __name__ == "__main__":
    tic(label="overall")    
    
    ###############################################
    ### data parameters
    ##
    ## Different LC-HRMS settings can be used per chromatogram. To be able to reuse them, they are summarized
    ##    in dictionaries. The keys are then used as the setting values
    ##
    ## polarities: specifies which filter lines are to be used for detecting the chromatographic peaks
    ## rtMaxDiffKNN: the maximum allowed difference to a k-nearest-neighbor in the retention time dimension in seconds
    ## ppmMaxDiffKNN: the maximum allowed difference to a k-nearest-neighbor in the m/z dimension
    ## nearestNeighbors: the number of nearest neighbors to consider
    ## rtMaxDiffKNN, ppmMaxDiffKNN and nearestNeighbors: these three parameters must be specified as lists
    ##                                                   each i-th entry will be used for a KNN alignment
    ## rtMaxDiffGrouping: the maximum allowed difference between features to be grouped into a group (in seconds)
    ## ppmMaxDiffGrouping: the maximuma allowed difference between features to be grouped into a group (in m/z units)
    ## rtWeight: the weight given to the retention time for calculating the distance between two features
    ## mzWeight: the weight given to the m/z for calculating the distance between two features
    ## fileTo: the files to which the grouped results shall be written (without an extension)
    ## file: all files belonging to the dataset
    expParams = {"WheatEar": {"polarities": {"positive": "Q Exactive (MS lvl: 1, pol: +)", "negative": "Q Exactive (MS lvl: 1, pol: -)"},
                              "rtMaxDiffKNN":[5, 5, 5], "ppmMaxDiffKNN":[15, 15, 15], "nearestNeighbors":[3, 3, 6], 
                              "rtMaxDiffGrouping": 2, "ppmMaxDiffGrouping": 5,
                              "rtWeight": 1, "mzWeight": 0.1,
                              "fileTo": "./Data/WheatEar",
                              "files": {"670_Sequence3_LVL1_1"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_1.mzXML"  },
                                        "670_Sequence3_LVL1_2"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_2.mzXML"  },
                                        "670_Sequence3_LVL1_3"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_3.mzXML"  },
                                        "670_Sequence3_LVL1x2_1": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_1.mzXML"},
                                        "670_Sequence3_LVL1x2_2": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_2.mzXML"},
                                        "670_Sequence3_LVL1x2_3": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_3.mzXML"},
                                        "670_Sequence3_LVL1x4_1": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_1.mzXML"},
                                        "670_Sequence3_LVL1x4_2": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_2.mzXML"},
                                        "670_Sequence3_LVL1x4_3": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_3.mzXML"},}},
                
                 "PHM": {"polarities": {"positive": "LTQ Orbitrap Velos (MS lvl: 1, pol: +)"},
                         "rtMaxDiffKNN":[5, 5, 5], "ppmMaxDiffKNN":[15, 15, 15], "nearestNeighbors":[3, 3, 6], 
                         "rtMaxDiffGrouping": 5, "ppmMaxDiffGrouping": 5,
                         "rtWeight": 1, "mzWeight": 0.1,
                         "fileTo": "./Data/PHM",
                         "files": {"05_EB3388_AOH_p_0" : {"file": "./Data/PHM/05_EB3388_AOH_p_0.mzXML" },
                                   "06_EB3389_AOH_p_10": {"file": "./Data/PHM/06_EB3389_AOH_p_10.mzXML"},
                                   "07_EB3390_AOH_p_20": {"file": "./Data/PHM/07_EB3390_AOH_p_20.mzXML"},
                                   "08_EB3391_AOH_p_60": {"file": "./Data/PHM/08_EB3391_AOH_p_60.mzXML"},
                                   "16_EB3392_AME_p_0" : {"file": "./Data/PHM/16_EB3392_AME_p_0.mzXML" },
                                   "17_EB3393_AME_p_10": {"file": "./Data/PHM/17_EB3393_AME_p_10.mzXML"},
                                   "18_EB3394_AME_p_20": {"file": "./Data/PHM/18_EB3394_AME_p_20.mzXML"},
                                   "19_EB3395_AME_p_60": {"file": "./Data/PHM/19_EB3395_AME_p_60.mzXML"},}},
                 
                 "HT29": {"polarities": {"positive": "Q Exactive HF (MS lvl: 1, pol: +)"},
                          "rtMaxDiffKNN":[5, 5, 5], "ppmMaxDiffKNN":[15, 15, 15], "nearestNeighbors":[3, 3, 6], 
                          "rtMaxDiffGrouping": 5, "ppmMaxDiffGrouping": 5,
                          "rtWeight": 1, "mzWeight": 0.1,
                          "fileTo": "./Data/HT29",
                          "files": {"HT_SOL1_LYS_010_pos": {"file": "./Data/HT29/HT_SOL1_LYS_010_pos.mzXML"},
                                    "HT_SOL1_SUP_025_pos": {"file": "./Data/HT29/HT_SOL1_SUP_025_pos.mzXML"},
                                    "HT_SOL2_LYS_014_pos": {"file": "./Data/HT29/HT_SOL2_LYS_014_pos.mzXML"},
                                    "HT_SOL2_SUP_029_pos": {"file": "./Data/HT29/HT_SOL2_SUP_029_pos.mzXML"},
                                    "HT_SOL3_LYS_018_pos": {"file": "./Data/HT29/HT_SOL3_LYS_018_pos.mzXML"},
                                    "HT_SOL3_LYS_033_pos": {"file": "./Data/HT29/HT_SOL3_SUP_033_pos.mzXML"},}},
                }
    

    ###############################################
    ## GPU information
    ##
    ## These values specify how the GPU is used for generating the training examples
    ## Please consult the documentation of your GPU.
    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
    blockdim = 512
    griddim  = 256
    
    
    ###############################################
    ## Finisehd with specifying LC-HRMS chromatogram files and LC-HRMS settings
    ## Nothing to change from here on
    

    for expName in expParams.keys():
        for polarity, filterLine in expParams[expName]["polarities"].items():
            tic("experiment")
            print("Processing experiment '%s', polarity '%s'"%(expName, polarity))
    
            ###############################################
            ### Iterate files and polarities (if FPS is used)
            features = None
            fileMapping = {}
            for inFile, fileProps in expParams[expName]["files"].items():

                ###############################################
                ### data parameters for chromatograms

                print("Loading sample '%s'"%(inFile))

                tsvFile = "%s_%sPeakBot.tsv"%(fileProps["file"].replace(".mzXML", ""), polarity)
                headers, rows = peakbot.readTSVFile(tsvFile, convertToMinIfPossible = True)
                print("  | .. %d features have been detected"%(len(rows)))

                a = len(fileMapping)
                fileMapping[a] = inFile
                headers.insert(0, "file")
                headers.append("featureID")
                headers.append("use")
                for row in rows:
                    row.insert(0, a)
                    row.append(-1)
                    row.append(-1)

                if False:
                    use = []

                    for rowi, row in enumerate(rows):
                        #if 495 < row[2] < 510 and 445 < row[3] < 458:
                        if 615 < row[2] < 629 and 261.2 < row[3] < 261.3:
                            use.append(rowi)

                    if len(use) > 0:
                        rows = [rows[i] for i in use]

                if len(rows) > 0:
                    if features is None:
                        features = np.matrix(rows)
                    else:
                        features = np.vstack((features, np.matrix(rows)))

                print("")

            print("A total of %d features have been detected"%(features.shape[0]))

            featuresOri = features.copy()
            features = peakbot.cuda.KNNalignFeatures(features, featuresOri, 
                                                     expParams[expName]["rtMaxDiffKNN"], expParams[expName]["ppmMaxDiffKNN"], expParams[expName]["nearestNeighbors"], 
                                                     expParams[expName]["rtWeight"], expParams[expName]["mzWeight"], 
                                                     blockdim = blockdim, griddim = griddim)
            headers, features = peakbot.cuda.groupFeatures(features, featuresOri, expParams[expName]["rtMaxDiffGrouping"], expParams[expName]["ppmMaxDiffGrouping"], fileMapping, blockdim = blockdim, griddim = griddim)

            peakbot.exportGroupedFeaturesAsFeatureML(headers, features, expParams[expName]["fileTo"]+"_"+polarity+"detectedFeatures.featureML")
            peakbot.exportGroupedFeaturesAsTSV(headers, features, expParams[expName]["fileTo"]+"_"+polarity+"detectedFeatures.tsv")

            print(".. processing experiment took %.1f seconds"%(toc("experiment")))
            print("\n\n\n\n\n\n")
    
    print("processing all datasets took %.1f seconds"%(toc("overall")))