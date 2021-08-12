#!/usr/bin/env python
# coding: utf-8

## run in python >=3.9
## activate conda environment on jucuda
## conda activate python3.9

location = "JuCuda"  ## JuCuda, HomePC

# Imports
import os
import pickle
import math
import shutil
import uuid
import numpy as np
import pandas as pd
import plotnine as p9

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.python.client import device_lib 
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
tf.get_logger().setLevel('WARNING')




######################################################################
###
### Load and import PeakBot
##
#

import sys
sys.path.append(os.path.join("..", "peakbot", "src"))
import peakbot.train.cuda
import peakbot.Chromatogram
            
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog


def loadFile(path):
    tic()
    mzxml = None
    if os.path.exists(path):
        mzxml = peakbot.Chromatogram.Chromatogram()
        mzxml.parse_file(path)
        print("Imported chromatogram for '%s'"%(inFile))
    return mzxml
    

###############################################
### Process files 
##
if __name__ == "__main__":
    tic(label="overall")        
    
    ###############################################
    ### data parameters    
    expParams = {"WheatEar" : {"polarities": {"positive": "Q Exactive (MS lvl: 1, pol: +)"},
                               "noiseLevel":1E3, "minRT":150, "maxRT":2250, "RTpeakWidth":[8,120],
                               "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                               "minApexBorderRatio":4, "minIntensity":1E5},
              
                 "PHM": {"polarities": {"positive": "LTQ Orbitrap Velos (MS lvl: 1, pol: +)"},
                         "noiseLevel":1E3, "minRT":100, "maxRT":750, "RTpeakWidth":[4,120],
                         "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                         "minApexBorderRatio":4, "minIntensity":1E5}}

    ###############################################
    ### chromatograms to process
    inFiles = {
        "670_Sequence3_LVL1_1"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_1.mzXML" , "params": "WheatEar"},
        "670_Sequence3_LVL1_2"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_2.mzXML" , "params": "WheatEar"},

        "670_Sequence3_LVL1x2_1": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_1.mzXML" , "params": "WheatEar"},
        "670_Sequence3_LVL1x2_2": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_2.mzXML" , "params": "WheatEar"},

        "670_Sequence3_LVL1x4_1": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_1.mzXML" , "params": "WheatEar"},
        "670_Sequence3_LVL1x4_2": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_2.mzXML" , "params": "WheatEar"},
    }
    exFiles = {
        "670_Sequence3_LVL1_3"  : {"file": "./Data/WheatEar/670_Sequence3_LVL1_3.mzXML"   , "params": "WheatEar"},
        "670_Sequence3_LVL1x2_3": {"file": "./Data/WheatEar/670_Sequence3_LVL1x2_3.mzXML" , "params": "WheatEar"},
        "670_Sequence3_LVL1x4_3": {"file": "./Data/WheatEar/670_Sequence3_LVL1x4_3.mzXML" , "params": "WheatEar"},
    }
    extFiles = {
        "08_EB3391_AOH_p_60":  {"file": "./Data/PHM/08_EB3391_AOH_p_60.mzXML" , "params": "PHM"}
    }
    
    ###############################################
    ## GPU information
    blockdim = 256
    griddim  = 128
    examplesDir = ""
    peakBotModelFile = "./temp/PBmodel.model.h5"
    logDir = "./temp/logs"
    
    if location == "HomePC":
        blockdim = (256, 1)
        griddim  = (128, 1)
        examplesDir = "E:\\temp"
        
    if location == "JuCuda":
        blockdim = 256
        griddim  = 512
        examplesDir = "/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
        
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
        
    maxRTOffset = 5
    maxMZOffset = 10
    
    maxPopulation = 4
    intensityScales = 10
    randomnessFactor = 0.1
    
    ###############################################
    ### Generate train instances
        
    headers, wepeaksTrain  = peakbot.readTSVFile("./WheatEar_trainPeaks.tsv" , convertToMinIfPossible = True)
    headers, wepeaksVal    = peakbot.readTSVFile("./WheatEar_valPeaks.tsv"   , convertToMinIfPossible = True)
    headers, wewalls       = peakbot.readTSVFile("./WheatEar_walls.tsv"      , convertToMinIfPossible = True)
    headers, webackgrounds = peakbot.readTSVFile("./WheatEar_backgrounds.tsv", convertToMinIfPossible = True)

    headers, epeaksVal    = peakbot.readTSVFile("./PHM_valPeaks.tsv"   , convertToMinIfPossible = True)
    headers, ewalls       = peakbot.readTSVFile("./PHM_walls.tsv"      , convertToMinIfPossible = True)
    headers, ebackgrounds = peakbot.readTSVFile("./PHM_backgrounds.tsv", convertToMinIfPossible = True)

    dsProps = {
        "T"  : {"files": inFiles , "peaks": wepeaksTrain, "walls": wewalls, "backgrounds": webackgrounds, "n": math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*peakbot.Config.EPOCHS/len(inFiles)), "shuffleSteps": 1E5},
        "V"  : {"files": inFiles , "peaks": wepeaksVal  , "walls": wewalls, "backgrounds": webackgrounds, "n": math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*128/len(inFiles))                  , "shuffleSteps": 1E4},
        "iT" : {"files": exFiles , "peaks": wepeaksTrain, "walls": wewalls, "backgrounds": webackgrounds, "n": math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH+128/len(exFiles))                  , "shuffleSteps": 1E4},
        "iV" : {"files": exFiles , "peaks": wepeaksVal  , "walls": wewalls, "backgrounds": webackgrounds, "n": math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH+128/len(exFiles))                  , "shuffleSteps": 1E4},
        "eV" : {"files": extFiles, "peaks": epeaksVal   , "walls": ewalls , "backgrounds": ebackgrounds , "n": math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH+128/len(extFiles))                 , "shuffleSteps": 1E4}
    }
    
    runTimes = []
    
    tf.random.set_seed(2021)
    np.random.seed(2021)
    
    if True:
        tic("Generated training and validation instances")
        
        for ds in dsProps.keys():
            try:
                shutil.rmtree(os.path.join(examplesDir, ds))
            except:
                pass
            os.mkdir(os.path.join(examplesDir, ds))
        print("removed old training instances in '%s'"%(examplesDir))
        
        ###############################################
        ### Iterate files and polarities (if FPS is used)
        
        for ds in dsProps.keys():
            print("Processing dataset '%s'"%ds)
            print("")
            
            for inFile, fileProps in dsProps[ds]["files"].items():
                tic(label="sample")

                params = expParams[fileProps["params"]]

                ###############################################
                ### data parameters for chromatograms
                polarities = params["polarities"]
                intraScanMaxAdjacentSignalDifferencePPM = params["intraScanMaxAdjacentSignalDifferencePPM"]
                interScanMaxSimilarSignalDifferencePPM = params["interScanMaxSimilarSignalDifferencePPM"]
                RTpeakWidth = params["RTpeakWidth"]
                minApexBorderRatio = params["minApexBorderRatio"]
                minIntensity = params["minIntensity"]

                for polarity, filterLine in polarities.items():
                    print("Processing dataset '%s', sample '%s', polarity '%s'"%(ds, inFile, polarity))
                    print("")

                    ###############################################
                    ### Load chromatogram
                    tic()
                    mzxml = loadFile(fileProps["file"])
                    mzxml.keepOnlyFilterLine(filterLine)
                    print("Filtered mzXML file for %s scan events only"%(polarity))
                    print("  | .. took %.1f seconds"%(toc()))
                    print("")

                    ###############################################
                    ### Generate train data
                    peakbot.train.cuda.generateTestInstances(mzxml, "'%s':'%s'"%(inFile, filterLine), 
                                                             dsProps[ds]["peaks"], dsProps[ds]["walls"], dsProps[ds]["backgrounds"],
                                                             nTestExamples = dsProps[ds]["n"], exportPath = os.path.join(examplesDir, ds), 
                                                             intraScanMaxAdjacentSignalDifferencePPM=intraScanMaxAdjacentSignalDifferencePPM, 
                                                             interScanMaxSimilarSignalDifferencePPM=interScanMaxSimilarSignalDifferencePPM, 

                                                             updateToLocalPeakProperties = True,
                                                             RTpeakWidth = RTpeakWidth, minApexBorderRatio = minApexBorderRatio, minIntensity = minIntensity, 
                                                             maxRTOffset = maxRTOffset, maxMZOffset = maxMZOffset,

                                                             maxPopulation = maxPopulation, intensityScales = intensityScales, randomnessFactor = randomnessFactor, 
                                                             blockdim = blockdim, griddim = griddim,
                                                             verbose = True)
                
            print("\n\n\n\n\n")        
        
        peakbot.train.shuffleResultsSampleNames(os.path.join(examplesDir, ds), verbose = True)
        peakbot.train.shuffleResults(os.path.join(examplesDir, ds), steps = dsProps[ds]["shuffleSteps"], samplesToExchange = 50, verbose = True)

    tocP("Generated training and validation instances", label="Generated training and validation instances")
    runTimes.append("Generating new training instances took %.1f seconds"%toc("Generated training and validation instances"))
    print("\n\n\n\n\n")
                        
    
    


    ###############################################
    ### Train new PeakBot Model
    histAll = None
    tic("train new PeakBot model")
    pb = None
    with strategy.scope():

        addValDS = []
        for ds in dsProps.keys():
            addValDS.append({"folder": os.path.join(examplesDir, ds), "name": ds, "numBatches": 128})

        pb, hist = peakbot.trainPeakBotModel(trainInstancesPath = os.path.join(examplesDir, "T"), 
                                             addValidationInstances = addValDS,
                                             logBaseDir = logDir, 
                                             verbose = True)

        pb.saveModelToFile(peakBotModelFile)
        print("Newly trained peakbot saved to file '%s'"%(peakBotModelFile))

        if histAll is None:
            histAll = hist
        else:
            hist = histAll.append(hist, ignore_index=True)

        print("")
        print("")

    histAll.to_pickle(os.path.join(".", "history_all.pandas.pickle"))
    tocP("train new PeakBot model","train new PeakBot model")

    df = pd.read_pickle(os.path.join(".", "history_all.pandas.pickle"))
    df['ID'] = df.model.str.split('_').str[-1]
    df = df[df["metric"]!="loss"]
    df.to_csv("./summaryStats.tsv", sep="\t", index=False)
    print(df)

    plot = (p9.ggplot(df, p9.aes("ID", "value", color="metric", group="metric"))
            + p9.facet_grid(".~set", scales="free_y")
            + p9.geom_point()
            + p9.geom_line()
            + p9.ggtitle("Replicates of selected model"))
    p9.options.figure_size = (19,8)
    p9.ggsave(plot=plot, filename="./summaryStats.png", height=7, width=12)
        
    runTimes.append("Traing a new PeakBot model took %.1f seconds"%toc("train new PeakBot model"))
    
    for r in runTimes:
        print(r)