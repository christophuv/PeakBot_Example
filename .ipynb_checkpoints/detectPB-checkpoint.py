#!/usr/bin/env python
# coding: utf-8

## run in python >=3.9
## activate conda environment on jucuda
## conda activate python3.9

location = "JuCuda"  ## JuCuda, HomePC

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
import peakbot.numpy
            
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog


def loadFile(path):
    tic()
    mzxml = None
    if os.path.exists(path+".pickle"):
        with open(path+".pickle", "rb") as inF:
            mzxml = pickle.load(inF)
        print("Imported chromatogram.pickle for '%s'"%(inFile))
    else:
        mzxml = peakbot.Chromatogram.Chromatogram()
        mzxml.parse_file(path)
        with open(path+".pickle", "wb") as outF:
            pickle.dump(mzxml, outF)
        print("Imported chromatogram for '%s'"%(inFile))
    return mzxml





###############################################
### Process files 
##
if __name__ == "__main__":
    tic(label="overall")

    ###############################################
    ### chromatograms to process
    expParams = {"WheatEar" : {"polarities": {"positive": "Q Exactive (MS lvl: 1, pol: +)"},
                               "noiseLevel":1E3, "minRT":150, "maxRT":2250, "RTpeakWidth":[8,120], "SavitzkyGolayWindowPlusMinus": 3,
                               "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                               "minApexBorderRatio":4, "minIntensity":1E5},
              
                 "PHM": {"polarities": {"positive": "LTQ Orbitrap Velos (MS lvl: 1, pol: +)"},
                         "noiseLevel":1E3, "minRT":100, "maxRT":750, "RTpeakWidth":[4,120], "SavitzkyGolayWindowPlusMinus": 3,
                         "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                         "minApexBorderRatio":4, "minIntensity":1E5},
                 
                 "HT29": {"polarities": {"positive": "Q Exactive HF (MS lvl: 1, pol: +)"},
                          "noiseLevel":1E3, "minRT":30, "maxRT":680, "RTpeakWidth":[2,30], "SavitzkyGolayWindowPlusMinus": 2,
                          "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                          "minApexBorderRatio":4, "minIntensity":1E5},
                }
    
    inFiles = {
        "670_Sequence3_LVL1_1"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_1.mzXML"  , "params": "WheatEar"},
        "670_Sequence3_LVL1_2"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_2.mzXML"  , "params": "WheatEar"},
        "670_Sequence3_LVL1_3"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_3.mzXML"  , "params": "WheatEar"},
        "670_Sequence3_LVL1x2_1": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_1.mzXML", "params": "WheatEar"},
        "670_Sequence3_LVL1x2_2": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_2.mzXML", "params": "WheatEar"},
        "670_Sequence3_LVL1x2_3": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_3.mzXML", "params": "WheatEar"},
        "670_Sequence3_LVL1x4_1": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_1.mzXML", "params": "WheatEar"},
        "670_Sequence3_LVL1x4_2": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_2.mzXML", "params": "WheatEar"},
        "670_Sequence3_LVL1x4_3": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_3.mzXML", "params": "WheatEar"},
        
        "05_EB3388_AOH_p_0" : {"file": "./Data/PHM/05_EB3388_AOH_p_0.mzXML" , "params": "PHM"},
        "06_EB3389_AOH_p_10": {"file": "./Data/PHM/06_EB3389_AOH_p_10.mzXML", "params": "PHM"},
        "07_EB3390_AOH_p_20": {"file": "./Data/PHM/07_EB3390_AOH_p_20.mzXML", "params": "PHM"},
        "08_EB3391_AOH_p_60": {"file": "./Data/PHM/08_EB3391_AOH_p_60.mzXML", "params": "PHM"},
        "16_EB3392_AME_p_0" : {"file": "./Data/PHM/16_EB3392_AME_p_0.mzXML" , "params": "PHM"},
        "17_EB3393_AME_p_10": {"file": "./Data/PHM/17_EB3393_AME_p_10.mzXML", "params": "PHM"},
        "18_EB3394_AME_p_20": {"file": "./Data/PHM/18_EB3394_AME_p_20.mzXML", "params": "PHM"},
        "19_EB3395_AME_p_60": {"file": "./Data/PHM/19_EB3395_AME_p_60.mzXML", "params": "PHM"},
        
        "HT_SOL1_LYS_010_pos": {"file": "./Data/HT29/HT_SOL1_LYS_010_pos.mzXML", "params": "HT29"},
        "HT_SOL1_SUP_025_pos": {"file": "./Data/HT29/HT_SOL1_SUP_025_pos.mzXML", "params": "HT29"},
        "HT_SOL2_LYS_014_pos": {"file": "./Data/HT29/HT_SOL2_LYS_014_pos.mzXML", "params": "HT29"},
        "HT_SOL2_SUP_029_pos": {"file": "./Data/HT29/HT_SOL2_SUP_029_pos.mzXML", "params": "HT29"},
        "HT_SOL3_LYS_018_pos": {"file": "./Data/HT29/HT_SOL3_LYS_018_pos.mzXML", "params": "HT29"},
        "HT_SOL3_LYS_033_pos": {"file": "./Data/HT29/HT_SOL3_SUP_033_pos.mzXML", "params": "HT29"},
    }

    ###############################################
    ### general parameters
        
    blockdim = 256
    griddim  = 128
    exportBatchSize = 512
    
    if location == "JuCuda":
        blockdim = 512
        griddim  = 256
        exportBatchSize= 2048
        
    peakBotModelFile = "./temp/PBmodel.model.h5"        
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    


    ###############################################
    ### Iterate files and polarities (if FPS is used)
    for inFile, fileProps in inFiles.items():
        tic(label="sample")
        
        params = expParams[fileProps["params"]]

        ###############################################
        ### Load chromatogram
        tic("sample")
        tic()
        mzxml = loadFile(fileProps["file"])       
        
    
        ###############################################
        ### data parameters for chromatograms
        polarities = params["polarities"]
        noiseLevel = params["noiseLevel"]
        minRT = params["minRT"]
        maxRT = params["maxRT"]
        intraScanMaxAdjacentSignalDifferencePPM = params["intraScanMaxAdjacentSignalDifferencePPM"]
        interScanMaxSimilarSignalDifferencePPM = params["interScanMaxSimilarSignalDifferencePPM"]
        RTpeakWidth = params["RTpeakWidth"]
        SavitzkyGolayWindowPlusMinus = params["SavitzkyGolayWindowPlusMinus"]
        minApexBorderRatio = params["minApexBorderRatio"]
        minIntensity = params["minIntensity"]
        
        for polarity, filterLine in polarities.items():
            print("Processing sample '%s', polarity '%s'"%(inFile, polarity))
            with tempfile.TemporaryDirectory() as tmpdirname:
                tic("instance")
                
                
                
                ###############################################
                ### Preprocess chromatogram
                tic()
                mzxml.keepOnlyFilterLine(filterLine)
                print("Filtered mzXML file for %s scan events only\n  | .. took %.1f seconds"%(polarity, toc()))
                print("")
                
                tic()
                mzxml.removeBounds(minRT=minRT, maxRT=maxRT, minMZ=100)
                mzxml.removeNoise(noiseLevel)
                print("Removed noise (%g) and bounds\n  | .. took %.1f seconds"%(noiseLevel, toc()))
                print("")
                
                
                
                ###############################################
                ### Detect local maxima with peak-like shapes## CUDA-GPU
                tic(label="preProcessing")
                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
                        mzxml, "'%s':'%s'"%(inFile, filterLine), 
                        intraScanMaxAdjacentSignalDifferencePPM = intraScanMaxAdjacentSignalDifferencePPM,
                        interScanMaxSimilarSignalDifferencePPM = interScanMaxSimilarSignalDifferencePPM,
                        RTpeakWidth = RTpeakWidth,
                        SavitzkyGolayWindowPlusMinus = SavitzkyGolayWindowPlusMinus, 
                        minApexBorderRatio = minApexBorderRatio,
                        minIntensity = minIntensity,
                        exportPath = tmpdirname, 
                        exportLocalMaxima = "peak-like-shape", # "all", "localMaxima-with-mzProfile", "peak-like-shape"
                        exportBatchSize = exportBatchSize, 
                        blockdim = blockdim,
                        griddim  = griddim, 
                        verbose = True)
                print("")
                peakbot.exportLocalMaximaAsFeatureML("%s_%sLM.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity), peaks)
                
                
                ###############################################
                ### Detect peaks with PeakBot
                tic("PeakBotDetection")
                peaks = []
                with strategy.scope():
                    peaks, walls, backgrounds, errors = peakbot.runPeakBot(tmpdirname, peakBotModelFile)
                print("")
                
                
                
                ###############################################
                ### Postprocessing
                tic("postProcessing")
                peaks = peakbot.cuda.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
                                                 blockdim = blockdim,
                                                 griddim  = griddim, 
                                                 verbose = True)
                print("")
                
                
                
                ## Log features
                TabLog().addData("%s - %s"%(inFile, filterLine), "Features", len(peaks))
                TabLog().addData("%s - %s"%(inFile, filterLine), "Walls", walls)
                TabLog().addData("%s - %s"%(inFile, filterLine), "Errors", errors)
                peakbot.exportPeakBotResultsFeatureML(peaks, "%s_%sPeakBot.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity))
                peakbot.exportPeakBotResultsTSV(peaks, "%s_%sPeakBot.tsv"%(fileProps["file"].replace(".mzXML", ""), polarity))
                print("Exported PeakBot detected peaks..")
                print("")
                
                
                
                tocP("File '%s':'%s': Preprocessed, exported and predicted with PeakBot"%(inFile, filterLine), label="instance")
                TabLog().addData("%s - %s"%(inFile, filterLine), "time (sec)", "%.1f"%toc("instance"))
                print("\n\n\n\n\n")
                
                if False:
                    peakbot.exportAreasAsFigures(tmpdirname, ".", 
                                                 model=tf.keras.models.load_model(peakBotModelFile, 
                                                      custom_objects = {"iou": peakbot.iou,
                                                                        "recall": peakbot.recall, 
                                                                        "precision": peakbot.precision, 
                                                                        "specificity": peakbot.specificity, 
                                                                        "negative_predictive_value": peakbot.negative_predictive_value, 
                                                                        "f1": peakbot.f1, 
                                                                        "pF1": peakbot.pF1, 
                                                                        "pTPR": peakbot.pTPR, 
                                                                        "pFPR": peakbot.pFPR}), 
                                                 maxExport = 25, threshold=0.0001, expFrom = 1000)
                    import sys
                    sys.exit(0)
                
        
    tocP("overall", label="overall")
    print("")
    TabLog().print()
