#!/usr/bin/env python
# coding: utf-8

import sys
import os

def getUserInput(message, options={"yes": ["yes", "y"], "no": ["no", "n"]}, default = None):
    
    ok = False
    printErr = False
    foundOpt = None
    while not ok:
        if printErr:
            print("Error, please only provide one of the listed options")
        print(message)
        print("Options (select an answer and provide one of the strings that qualify for this answer): ")
        for opt in options.keys():
            print("  %s%s: %s"%(opt, "(default if you just press Enter)" if default == opt else "", ", ".join(s for s in options[opt])))
        value = input("Input: ")
        if value == "":
            value = default

        for opt in options.keys():
            if value in options[opt]:
                foundOpt = opt
                ok = True
        printErr = True

    return foundOpt


print('''

    This is a quick example for PeakBot
    -----------------------------------

    It will download the samples of the demonstration dataset MTBLS1358 (https://www.ebi.ac.uk/metabolights/MTBLS1358)
    and place them into the folder MTBLS1358 in the current directory. 

    Then it will generate the training and validation instances for a new PeakBot CNN model and 
    finally train a new model for detecting the chromatographic peaks. 

    Finally it will predict the chromatographic peaks in the samples. 


    To run it, please make sure that PeakBot has been installed. For instructions please refer to 
    https://github.com/christophuv/PeakBot
    

''')


print("Download dataset")
if getUserInput("Do you want to continue?") == "yes":
    
    ##################################################
    ### Doanload data
    ## This step can be omitted for other datasets
    ##

    import urllib.request
    import shutil

    files = {
        "HT_SOL1_LYS_010_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/kKgd5gBlIlsG9Bj/download",
        "HT_SOL1_SUP_025_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/42fEfzSUn4OLFDl/download",
        "HT_SOL2_LYS_014_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/84gnlkVxYdWnXC8/download",
        "HT_SOL2_SUP_029_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/fidRZt5EtJ1y57J/download",
        "HT_SOL3_LYS_018_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/O5QoSHBwl97QuV7/download",
        "HT_SOL3_SUP_033_pos.mzXML": "https://ucloud.univie.ac.at/index.php/s/RPyxKpP0RqEhHpm/download",
    }
    refs = {
        "Backgrounds.tsv": "https://ucloud.univie.ac.at/index.php/s/4IYdl3sdV92l8l3/download",
        "Peaks.tsv": "https://ucloud.univie.ac.at/index.php/s/UirxEIecOW55zkv/download",
        "Walls.tsv": "https://ucloud.univie.ac.at/index.php/s/XMMVhdHpsOdVVT9/download",
    }
    models = {
        "PBmodel_MTBLS1358.model.h5": "https://ucloud.univie.ac.at/index.php/s/8J0B6X9HxWx9qQ6/download"
    }

    try:
        shutil.rmtree(os.path.join(".", "MTBLS1358"))
    except:
        pass

    for a in [os.path.join(".","MTBLS1358"), 
              os.path.join(".","MTBLS1358", "Data"), 
              os.path.join(".","MTBLS1358", "Reference"),
              os.path.join(".","MTBLS1358", "Model"), 
              os.path.join(".","MTBLS1358", "Temp")]:
        try:
            os.mkdir(a)
        except:
            pass

    print("Downloading data files")
    for fileName, url in files.items():
        print("  ", os.path.join(".","MTBLS1358", "Data", fileName))
        urllib.request.urlretrieve(url, os.path.join(".","MTBLS1358", "Data", fileName))
    print("Downloading reference files")
    for fileName, url in refs.items():
        print("  ", os.path.join(".","MTBLS1358", "Reference", fileName))
        urllib.request.urlretrieve(url, os.path.join(".","MTBLS1358", "Reference", fileName))
    print("Downloading pre-trained model")
    for fileName, url in models.items():
        print("  ", os.path.join(".","MTBLS1358", "Model", fileName))
        urllib.request.urlretrieve(url, os.path.join(".","MTBLS1358", "Model", fileName))
print("\n\n\n\n\n")













##################################################
### Parameter definition
## 

## Specific tensorflow configuration. Can re omitted
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf


###############################################
### Name of the experiment
##
## expNAme: specifies the name of the experiment and the folder in which the data will be processed
expName = "MTBLS1358"


###############################################
### data parameters
##
## Different LC-HRMS settings can be used per chromatogram. To be able to reuse them, they are summarized
##    in dictionaries. The keys are then used as the setting values
## 
## PeakBotModel: file to save the new PeakBot CNN model to
## polarities: specifies which filter lines are to be used for detecting the chromatographic peaks
## noiseLevel: Everything below this threshold is considered noise and removed directly after the import
## minRT / maxRT: Area of the chromatogram in which chromatographic peaks are expected
## RTpeakWidth: array of [minimum, maximum] peak-width in scans
## intraScanMaxAdjacentSignalDifferencePPM: Maximum difference of signals belonging to the same profile mode peak
## interScanMaxSimilarSignalDifferencePPM: Maximum difference of signals representing the same profile mode signal
## minIntensity: All signals below this threshold are not considered for the local maximum detection
expParams = {"PeakBotModel": os.path.join(".", expName, "Model", "PBmodel_MTBLS1358.model.h5"),
             "polarities": {"positive": "Q Exactive HF (MS lvl: 1, pol: +)"},
             "minRT": 30, "maxRT": 680, "RTpeakWidth":[2,30], "SavitzkyGolayWindowPlusMinus": 3,
             "intraScanMaxAdjacentSignalDifferencePPM": 15, "interScanMaxSimilarSignalDifferencePPM": 3,
             "noiseLevel": 1E3, "minIntensity": 1E5}

###############################################
### chromatograms to process for training/validation
inFiles = {        
    "HT_SOL1_LYS_010_pos": os.path.join(".", expName, "Data", "HT_SOL1_LYS_010_pos.mzXML"),
    "HT_SOL2_LYS_014_pos": os.path.join(".", expName, "Data", "HT_SOL2_LYS_014_pos.mzXML"),
}
exFiles = {
    "HT_SOL3_LYS_018_pos": os.path.join(".", expName, "Data", "HT_SOL3_LYS_018_pos.mzXML"),
    #"HT_SOL3_SUP_033_pos": os.path.join(".", expName, "Data", "HT_SOL3_SUP_033_pos.mzXML"), ## Reference peaks are not taken from SUP samples, thus they should not be used here
}


###############################################
### chromatograms to process during detection
detFiles = {
    "HT_SOL1_LYS_010_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL1_LYS_010_pos.mzXML"),
    "HT_SOL1_SUP_025_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL1_SUP_025_pos.mzXML"),
    "HT_SOL2_LYS_014_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL2_LYS_014_pos.mzXML"),
    "HT_SOL2_SUP_029_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL2_SUP_029_pos.mzXML"),
    "HT_SOL3_LYS_018_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL3_LYS_018_pos.mzXML"),
    "HT_SOL3_LYS_033_pos_cm" : os.path.join(".", expName, "Data", "HT_SOL3_SUP_033_pos.mzXML"),
}

###############################################
## GPU information
##
## These values specify how the GPU is used for generating the training examples
## Please consult the documentation of your GPU.
## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64, exportBatchSize = 1024
## Values for a high-end HPC Nvidia Tesla V100S card with 32GB GPU-memory are blockdim = 16, griddim = 512, exportBatchSize = 12288
## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
## The strategy specifies on which device tensorflow shall be executed.
exportBatchSize = 1024
strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")

###############################################
## Values for generating the training instances
##
## maxPopulation: This specifies how many different references (features, backgrounds) shall be combined in a
##   single training instance. The values specifies the maximum number and values between 1 and maxPopulation
##   will be picked randomly
## intensityScales: Specifies the factor with which the references will be scales (1/intensityScales to intensityScales)
## randomnessFactor: Specifies the factor which which the individual signals of the raw data are multiplied
maxPopulation = 4
intensityScales = 10
randomnessFactor = 0.1




# Imports
import os
import pickle
import math
import shutil
import random
import tempfile
import numpy as np
import pandas as pd
import plotnine as p9

## Load the PeakBot package
import sys
sys.path.append(os.path.join("..", "peakbot", "src"))
import peakbot
import peakbot.train.cpu
import peakbot.Chromatogram
from peakbot.core import tic, toc, tocP, TabLog


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
























print("Do you want to use the available pre-trained model?")
if getUserInput("Use pre-trained model? (recommended for testing and new users)") == "yes":
    pass 
elif getUserInput("Do you want to train a new model") == "yes":

    print("Train new model (saving to '%s', this might take long depending on your computer hardware)"%(expParams["PeakBotModel"]))
    if input("Do you want to continue? (y[es]/no)").lower() in ["yes", "y"]:

        ##################################################
        ### Train new model
        ##
        print("Training instances will be generated and a new PeakBot CNN model will be trained")


        tic(label="overall")



        ###############################################
        ### Generate train instances
        ##
        ## The different training sets are loaded from the files
        ## Different references an be loaded for different training and validation datastes
        ## Finally, all training and validation datasets are compiled into different sets in the variable dsProps
        ##    For each such dataset the chromatograms, reference peaks, backgrounds and walls must be specified as well
        ##    as the number of instances to be generated
        headers, peaks       = peakbot.readTSVFile(os.path.join(".", expName, "Reference", "Peaks.tsv"      ), convertToMinIfPossible = True)
        headers, walls       = peakbot.readTSVFile(os.path.join(".", expName, "Reference", "Walls.tsv"      ), convertToMinIfPossible = True)
        headers, backgrounds = peakbot.readTSVFile(os.path.join(".", expName, "Reference", "Backgrounds.tsv"), convertToMinIfPossible = True)
        random.shuffle(peaks)
        a = int(len(peaks)*0.6)
        peaksTrain = peaks[:a]
        peaksVal   = peaks[a:]
        print("Using %d peaks for training and %d peaks for internal validation"%(a, len(peaks)-a))

        dsProps = {
            "T"  : {"files": inFiles , "peaks": peaksTrain, "walls": walls, "backgrounds": backgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*peakbot.Config.EPOCHS/len(inFiles))), "shuffleSteps": 1E4},
            "V"  : {"files": inFiles , "peaks": peaksVal  , "walls": walls, "backgrounds": backgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(inFiles)))                    , "shuffleSteps": 1E4},
            "iT" : {"files": exFiles , "peaks": peaksTrain, "walls": walls, "backgrounds": backgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(exFiles)))                    , "shuffleSteps": 1E4},
            "iV" : {"files": exFiles , "peaks": peaksVal  , "walls": walls, "backgrounds": backgrounds, "n": max(2**14,math.ceil(peakbot.Config.BATCHSIZE*peakbot.Config.STEPSPEREPOCH*8/len(exFiles)))                    , "shuffleSteps": 1E4},
        }

        ###############################################
        ### Generate training instances from the previously specified training and validation datasets
        ## (no changes are required here)
        runTimes = []

        ## The random seeds are set
        tf.random.set_seed(2021)
        np.random.seed(2021)

        histAll = None
        try:
            os.remove(os.path.join(".", expName, "Temp", "History_MTBLS1358.pandas.pickle"))
        except Exception:
            pass

        with tempfile.TemporaryDirectory() as examplesDir:
            tic("Generated training and validation instances")
            for ds in dsProps.keys():
                print("Processing dataset '%s'"%ds)
                print("")

                os.mkdir(os.path.join(examplesDir, ds))

                ###############################################
                ### Iterate files and polarities (if FPS is used)
                ## (no changes are required here)
                for inFile, fileLoc in dsProps[ds]["files"].items():
                    tic(label="sample")

                    ###############################################
                    ### Data parameters for chromatograms
                    polarities = expParams["polarities"]
                    intraScanMaxAdjacentSignalDifferencePPM = expParams["intraScanMaxAdjacentSignalDifferencePPM"]
                    interScanMaxSimilarSignalDifferencePPM = expParams["interScanMaxSimilarSignalDifferencePPM"]
                    RTpeakWidth = expParams["RTpeakWidth"]
                    minIntensity = expParams["minIntensity"]

                    for polarity, filterLine in polarities.items():
                        print("Processing chromatogram '%s', sample '%s', polarity '%s'"%(ds, inFile, polarity))
                        print("")

                        ###############################################
                        ### Load chromatogram
                        tic()
                        mzxml = loadFile(fileLoc)
                        print("Available filter lines for file '%s': %s"%(inFile, str(mzxml.getFilterLinesPerPolarity())))
                        mzxml.keepOnlyFilterLine(filterLine)
                        print("Filtered chromatogram file for %s scan events only"%(polarity))
                        print("")

                        ###############################################
                        ### Generate train data
                        peakbot.train.cpu.generateTestInstances(
                            mzxml, "'%s':'%s'"%(inFile, filterLine),
                            dsProps[ds]["peaks"], dsProps[ds]["walls"], dsProps[ds]["backgrounds"],

                            nTestExamples = dsProps[ds]["n"], exportPath = os.path.join(examplesDir, ds),

                            intraScanMaxAdjacentSignalDifferencePPM=intraScanMaxAdjacentSignalDifferencePPM,
                            interScanMaxSimilarSignalDifferencePPM=interScanMaxSimilarSignalDifferencePPM,
                            updateToLocalPeakProperties = True,

                            RTpeakWidth = RTpeakWidth, minIntensity = minIntensity,

                            maxPopulation = maxPopulation, intensityScales = intensityScales, randomnessFactor = randomnessFactor,

                            verbose = True)

                ###############################################
                ### Shuffle generated training/validation dataset from the different chromatograms
                peakbot.train.shuffleResultsSampleNames(os.path.join(examplesDir, ds), verbose = True)
                peakbot.train.shuffleResults(os.path.join(examplesDir, ds), steps = dsProps[ds]["shuffleSteps"], samplesToExchange = 50, verbose = True)

            tocP("Generated training and validation instances", label="Generated training and validation instances")
            runTimes.append("Generating new training/validation instances for the datasets took %.1f seconds"%(toc("Generated training and validation instances")))
            print("\n\n\n\n\n")

            

            ###############################################
            ### Train new PeakBot Model
            ## (no changes are required here)
            tic("train new PeakBot model")
            pb = None
            with strategy.scope():

                addValDS = []
                for ds in dsProps.keys():
                    addValDS.append({"folder": os.path.join(examplesDir, ds), "name": ds, "numBatches": 512})

                pb, hist = peakbot.trainPeakBotModel(trainInstancesPath = os.path.join(examplesDir, "T"),
                                                     addValidationInstances = addValDS,
                                                     logBaseDir = os.path.join(".", expName, "Temp"),
                                                     verbose = True)

                pb.saveModelToFile(expParams["PeakBotModel"])
                print("Newly trained peakbot saved to file '%s'"%(expParams["PeakBotModel"]))


                if histAll is None:
                    histAll = hist
                else:
                    histAll = histAll.append(hist, ignore_index=True)

                print("")
                print("")

                ### Summarize the training and validation metrices and losses
                ## (no changes are required here)
                histAll.to_pickle(os.path.join(".", expName, "Temp", "History_MTBLS1358.pandas.pickle"))
                tocP("train new PeakBot model","train new PeakBot model")
                runTimes.append("Traing a new PeakBot model took %.1f seconds"%toc("train new PeakBot model"))

            
            
        ###############################################
        ### Summarize and illustrate the results of the different training and validation dataset
        ## (no changes are required here)
        df = pd.read_pickle(os.path.join(".", expName, "Temp", "History_MTBLS1358.pandas.pickle"))
        df['ID'] = df.model.str.split('_').str[-1]
        df = df[df["metric"]!="loss"]
        df = df[df["set"]!="eV"]
        df.to_csv(os.path.join(".", expName, "Temp", "SummaryPlot_MTBLS1358.tsv"), sep="\t", index=False)

        plot = (p9.ggplot(df, p9.aes("set", "value", colour="set"))
                + p9.geom_point()
                + p9.facet_wrap("~metric", scales="free_y", ncol=2)
                + p9.scale_x_discrete(limits=["T", "V", "iT", "iV"])
                + p9.ggtitle("Training losses/metrics") + p9.xlab("Training/Validation dataset") + p9.ylab("Value")
                + p9.theme(legend_position = "none", panel_spacing_x=0.5))
        p9.options.figure_size = (5.2,5)
        p9.ggsave(plot=plot, filename=os.path.join(".", expName, "Temp", "SummaryPlot_MTBLS1358.png"), width=5.2, height=5, dpi=300)


        ## Print runtimes
        for r in runTimes:
            print(r)
print("\n\n\n\n\n")



















##################################################
### Detect chromatographic peaks
##
print("The trained model (loading from '%s') will be used to detect chromatographic peaks in LC-HRMS chromatograms"%(expParams["PeakBotModel"]))
TabLog().reset()
tic(label="overall")

###############################################
## Finisehd with specifying LC-HRMS chromatogram files and LC-HRMS settings
## Nothing to change from here on


###############################################
### Iterate files and polarities (if FPS is used)
for inFile, fileLoc in detFiles.items():
    tic(label="sample")
    
    for polarity, filterLine in expParams["polarities"].items():
        print("Processing sample '%s', polarity '%s'"%(inFile, polarity))
        with tempfile.TemporaryDirectory() as tmpdirname:
            tic("instance")
            
            ###############################################
            ### Preprocess chromatogram
            tic("sample")
            mzxml = loadFile(fileLoc)
            mzxml.keepOnlyFilterLine(filterLine)
            print("  | .. filtered mzXML file for %s scan events only"%(polarity))
            mzxml.removeBounds(minRT = expParams["minRT"], maxRT = expParams["maxRT"])
            mzxml.removeNoise(expParams["noiseLevel"])
            print("  | .. removed noise and bounds")
            print("")
            
            ###############################################
            ### Detect local maxima with peak-like shapes## CUDA-GPU
            tic(label="preProcessing")
            peaks, maximaProps, maximaPropsAll = peakbot.cpu.preProcessChromatogram(
                    mzxml, "'%s':'%s'"%(inFile, filterLine), 
                    intraScanMaxAdjacentSignalDifferencePPM = expParams["intraScanMaxAdjacentSignalDifferencePPM"],
                    interScanMaxSimilarSignalDifferencePPM = expParams["interScanMaxSimilarSignalDifferencePPM"],
                    RTpeakWidth = expParams["RTpeakWidth"],
                    SavitzkyGolayWindowPlusMinus = expParams["SavitzkyGolayWindowPlusMinus"], 
                    minIntensity = expParams["minIntensity"],
                    exportPath = tmpdirname, 
                    exportLocalMaxima = "peak-like-shape", # "all", "localMaxima-with-mzProfile", "peak-like-shape"
                    exportBatchSize = exportBatchSize, 
                    verbose = True)
            print("")
            TabLog().addData("%s - %s"%(inFile, filterLine), "LM", len(peaks))
            
            ###############################################
            ### Detect peaks with PeakBot
            tic("PeakBotDetection")
            peaks = []
            with strategy.scope():
                peaks, walls, backgrounds, errors = peakbot.runPeakBot(tmpdirname, expParams["PeakBotModel"])
            print("")
            
            ###############################################
            ### Postprocessing
            tic("postProcessing")
            peaks = peakbot.cpu.postProcess(mzxml, "'%s':'%s'"%(inFile, filterLine), peaks, 
                                                verbose = True)
            print("")
            
            ## Log features
            TabLog().addData("%s - %s"%(inFile, filterLine), "Features", len(peaks))
            TabLog().addData("%s - %s"%(inFile, filterLine), "Walls", len(walls))
            TabLog().addData("%s - %s"%(inFile, filterLine), "Backgrounds", len(backgrounds))
            TabLog().addData("%s - %s"%(inFile, filterLine), "Errors", len(errors))
            peakbot.exportPeakBotResultsFeatureML(peaks, "%s_%sPeakBot.featureML"%(fileLoc.replace(".mzXML", ""), polarity))
            peakbot.exportPeakBotResultsTSV(peaks, "%s_%sPeakBot.tsv"%(fileLoc.replace(".mzXML", ""), polarity))
            print("Exported PeakBot detected peaks..")
            print("")
            
            tocP("File '%s':'%s': Preprocessed, exported and predicted with PeakBot"%(inFile, filterLine), label="instance")
            TabLog().addData("%s - %s"%(inFile, filterLine), "time (sec)", "%.1f"%toc("instance"))
            print("\n\n\n\n\n")                
    
TabLog().addData("Total time all files", "time (sec)", "%.1f"%toc("overall"))
print("")
TabLog().print()
print("\n\n\n\n\n")
