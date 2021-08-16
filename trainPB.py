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

## Specific tensorflow configuration. Can re omitted
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

## Import PeakBot from directory not as installed package (can be omitted if the package is installed)
import sys
sys.path.append(os.path.join("..", "peakbot", "src"))

## Load the PeakBot package
import peakbot.train.cuda
import peakbot.Chromatogram
from peakbot.core import tic, toc, tocP, tocAddStat, addFunctionRuntime, timeit, printRunTimesSummary, TabLog



## Function for loading mzXML files (and saving them as pickle file for fast access)
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
    ##
    ## Different LC-HRMS settings can be used per chromatogram. To be able to reuse them, they are summarized
    ##    in dictionaries. The keys are then used as the setting values
    ##
    ## polarities: specifies which filter lines are to be used for detecting the chromatographic peaks
    ## noiseLevel: Everything below this threshold is considered noise and removed directly after the import
    ## minRT / maxRT: Area of the chromatogram in which chromatographic peaks are expected
    ## RTpeakWidth: array of [minimum, maximum] peak-width in scans
    ## intraScanMaxAdjacentSignalDifferencePPM: Maximum difference of signals belonging to the same profile mode peak
    ## interScanMaxSimilarSignalDifferencePPM: Maximum difference of signals representing the same profile mode signal
    ## minIntensity: All signals below this threshold are not considered for the local maximum detection
    expParams = {"WheatEar" : {"polarities": {"positive": "Q Exactive (MS lvl: 1, pol: +)"},
                               "noiseLevel":1E3, "minRT":150, "maxRT":2250, "RTpeakWidth":[8,120],
                               "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                               "minIntensity":1E5},

                 "PHM": {"polarities": {"positive": "LTQ Orbitrap Velos (MS lvl: 1, pol: +)"},
                         "noiseLevel":1E3, "minRT":100, "maxRT":750, "RTpeakWidth":[4,120],
                         "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                         "minIntensity":1E5}}

    ###############################################
    ### chromatograms to process
    ##
    ## Different LC-HRMS chromatograms can be used for generating a training or validation dataset
    ##
    ## file: Path of the mzXML file
    ## params: parameter collection for the particular sample (see variable expParams)
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
    ##
    ## These values specify how the GPU is used for generating the training examples
    ## Please consult the documentation of your GPU.
    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
    ## The strategy specifies on which device tensorflow shall be executed.
    blockdim = 256
    griddim  = 128
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")

    ###############################################
    ## Temporary directories
    ##
    ## examplesDir should point to an existing empty directory with at least 50GB free space
    examplesDir = "C:\\temp"#/home/users/cbueschl/_ScratchFromJuCUDA/burning_scratch/cbueschl/examples/CUDA"
    peakBotModelFile = "./temp/PBmodel.model.h5"
    logDir = "./temp/logs"

    ###############################################
    ## Values for generating the training instances
    ##
    ## maxPopulation: This specifies how many different references (features, backgrounds) shall be combined in a
    ##   single training instance. The values specifies the maximum number and values between 1 and maxPopulation
    ##   will be picked randomly
    ## intensityScales: Specifies the factor with which the references will be scales (1/intensityScales to intensityScales)
    maxPopulation = 4
    intensityScales = 10
    randomnessFactor = 0.1

    ###############################################
    ### Generate train instances
    ##
    ## The different training sets are loaded from the files
    ## Different references an be loaded for different training and validation datastes
    ## Finally, all training and validation datasets are compiled into different sets in the variable dsProps
    ##    For each such dataset the chromatograms, reference peaks, backgrounds and walls must be specified as well
    ##    as the number of instances to be generated
    headers, wepeaksTrain  = peakbot.readTSVFile("./Reference/WheatEar_trainPeaks.tsv" , convertToMinIfPossible = True)
    headers, wepeaksVal    = peakbot.readTSVFile("./Reference/WheatEar_valPeaks.tsv"   , convertToMinIfPossible = True)
    headers, wewalls       = peakbot.readTSVFile("./Reference/WheatEar_walls.tsv"      , convertToMinIfPossible = True)
    headers, webackgrounds = peakbot.readTSVFile("./Reference/WheatEar_backgrounds.tsv", convertToMinIfPossible = True)

    headers, epeaksVal    = peakbot.readTSVFile("./Reference/PHM_valPeaks.tsv"   , convertToMinIfPossible = True)
    headers, ewalls       = peakbot.readTSVFile("./Reference/PHM_walls.tsv"      , convertToMinIfPossible = True)
    headers, ebackgrounds = peakbot.readTSVFile("./Reference/PHM_backgrounds.tsv", convertToMinIfPossible = True)

    dsProps = {
        "T"  : {"files": inFiles , "peaks": wepeaksTrain, "walls": wewalls, "backgrounds": webackgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*peakbot.Config.EPOCHS/len(inFiles))), "shuffleSteps": 1E5},
        "V"  : {"files": inFiles , "peaks": wepeaksVal  , "walls": wewalls, "backgrounds": webackgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(inFiles)))                  , "shuffleSteps": 1E4},
        "iT" : {"files": exFiles , "peaks": wepeaksTrain, "walls": wewalls, "backgrounds": webackgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(exFiles)))                  , "shuffleSteps": 1E4},
        "iV" : {"files": exFiles , "peaks": wepeaksVal  , "walls": wewalls, "backgrounds": webackgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(exFiles)))                  , "shuffleSteps": 1E4},
        "eV" : {"files": extFiles, "peaks": epeaksVal   , "walls": ewalls , "backgrounds": ebackgrounds , "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(extFiles)))                 , "shuffleSteps": 1E4}
    }





    ###############################################
    ### Generate training instances from the previously specified training and validation datasets
    ## (no changes are required here)
    runTimes = []

    ## The random seeds are set
    tf.random.set_seed(2021)
    np.random.seed(2021)

    if True:
        histAll = None
        try:
            os.remove(os.path.join(".", "Data", "history_all.pandas.pickle"))
        except Exception:
            pass
        tic("Generated training and validation instances")

        for ds in dsProps.keys():
            try:
                shutil.rmtree(os.path.join(examplesDir, ds))
            except:
                pass
            os.mkdir(os.path.join(examplesDir, ds))
            print("removed old training instances in '%s'"%(os.path.join(examplesDir, ds)))

            ###############################################
            ### Iterate files and polarities (if FPS is used)
            ## (no changes are required here)
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
                    peakbot.train.cuda.generateTestInstances(
                        mzxml, "'%s':'%s'"%(inFile, filterLine),
                        dsProps[ds]["peaks"], dsProps[ds]["walls"], dsProps[ds]["backgrounds"],

                        nTestExamples = dsProps[ds]["n"], exportPath = os.path.join(examplesDir, ds),

                        intraScanMaxAdjacentSignalDifferencePPM=intraScanMaxAdjacentSignalDifferencePPM,
                        interScanMaxSimilarSignalDifferencePPM=interScanMaxSimilarSignalDifferencePPM,
                        updateToLocalPeakProperties = True,

                        RTpeakWidth = RTpeakWidth, minIntensity = minIntensity,

                        maxPopulation = maxPopulation, intensityScales = intensityScales, randomnessFactor = randomnessFactor,

                        blockdim = blockdim, griddim = griddim,
                        verbose = True)

            print("\n\n\n\n\n")


            ###############################################
            ### data parameters for chromatograms
            peakbot.train.shuffleResultsSampleNames(os.path.join(examplesDir, ds), verbose = True)
            peakbot.train.shuffleResults(os.path.join(examplesDir, ds), steps = dsProps[ds]["shuffleSteps"], samplesToExchange = 50, verbose = True)

            tocP("Generated training and validation instances", label="Generated training and validation instances")
            runTimes.append("Generating new training instances took %.1f seconds"%toc("Generated training and validation instances"))
            print("\n\n\n\n\n")




    if True:
        ###############################################
        ### Train new PeakBot Model
        ## (no changes are required here)
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
                histAll = histAll.append(hist, ignore_index=True)

            print("")
            print("")

            ### Summarize the training and validation metrices and losses
            ## (no changes are required here)
            histAll.to_pickle(os.path.join(".", "Data", "history_all.pandas.pickle"))
            tocP("train new PeakBot model","train new PeakBot model")
            runTimes.append("Traing a new PeakBot model took %.1f seconds"%toc("train new PeakBot model"))


    df = pd.read_pickle(os.path.join(".", "Data", "history_all.pandas.pickle"))
    df['ID'] = df.model.str.split('_').str[-1]
    df = df[df["metric"]!="loss"]
    df.to_csv(os.path.join(".", "Data", "summaryStats.tsv"), sep="\t", index=False)
    print(df)

    plot = (p9.ggplot(df, p9.aes("ID", "value", color="metric", group="metric"))
            + p9.facet_grid(".~set", scales="free_y")
            + p9.geom_point()
            + p9.geom_line()
            + p9.ggtitle("Replicates of selected model"))
    p9.options.figure_size = (19,8)
    p9.ggsave(plot=plot, filename=os.path.join(".", "Data", "summaryStats.png"), height=7, width=12)

    plot = (p9.ggplot(df, p9.aes("set", "value", colour="set"))
            + p9.geom_point()
            + p9.facet_wrap("~metric", scales="free_y", ncol=2)
            + p9.ggtitle("Training losses/metrics") + p9.xlab("Training/Validation dataset") + p9.ylab("Value")
            + p9.theme(legend_position = "none", panel_spacing_x=0.5))
    p9.options.figure_size = (5.2,5)
    p9.ggsave(plot=plot, filename=os.path.join(".", "Data", "summary_Stats_overview.png"), width=5.2, height=5, dpi=300)


    ## Print runtimes
    for r in runTimes:
        print(r)
