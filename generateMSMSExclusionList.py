#!/usr/bin/env python
# coding: utf-8

## run in python >= 3.8
## activate conda environment on jucuda
## conda activate python3.8

# Imports
import os
import pickle
import tempfile
from collections import OrderedDict

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

import sys
sys.path.append(os.path.join("..", "peakbot", "src"))
import peakbot
import peakbot.Chromatogram
import peakbot.cuda            
from peakbot.core import tic, toc, tocP, TabLog


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
    ### data parameters
    ##
    ## Different LC-HRMS settings can be used per chromatogram. To be able to reuse them, they are summarized
    ##    in dictionaries. The keys are then used as the setting values
    ##
    ## polarities: specifies which filter lines are to be used for detecting the chromatographic peaks
    ## noiseLevel: Everything below this threshold is considered noise and removed directly after the import
    ## minRT / maxRT: Area of the chromatogram in which chromatographic peaks are expected
    ## RTpeakWidth: array of [minimum, maximum] peak-width in scans
    ## SavitzkyGolayWindowPlusMinus: specifies the degree of smoothing. A value of x results in a smoothing window of 2*x + 1
    ## intraScanMaxAdjacentSignalDifferencePPM: Maximum difference of signals belonging to the same profile mode peak
    ## interScanMaxSimilarSignalDifferencePPM: Maximum difference of signals representing the same profile mode signal
    ## minIntensity: All signals below this threshold are not considered for the local maximum detection
    expParams = {"WheatEar": {"polarities": {"negative": "Q Exactive (MS lvl: 1, pol: -)", "positive": "Q Exactive (MS lvl: 1, pol: +)"},
                              "PeakBotModel": "./temp/PBmodel_WheatEar.model.h5",
                              "minRT":150, "maxRT":2250, "RTpeakWidth":[8,120], "SavitzkyGolayWindowPlusMinus": 3,
                              "intraScanMaxAdjacentSignalDifferencePPM":15, "interScanMaxSimilarSignalDifferencePPM":3,
                              "noiseLevel":1E3, "minIntensity":1E5},
                }

    ###############################################
    ### chromatograms to process
    ##
    ## Different LC-HRMS chromatograms can be used for generating a training or validation dataset
    ##
    ## file: Path of the mzXML file
    ## params: parameter collection for the particular sample (see variable expParams)
    inFiles = OrderedDict()

    # Unreleased data
    # Wheat ear (similar to untreated samples of: Stable Isotope–Assisted Plant Metabolomics: Combination of Global and Tracer-Based Labeling for Enhanced Untargeted Profiling and Compound Annotation)
    # https://doi.org/10.3389/fpls.2019.01366
    #inFiles["670_Sequence3_LVL1_1"    ] = {"file": "./Data/WheatEar/670_Sequence3_LVL1_1.mzXML"  , "params": "WheatEar"}
    inFiles["823_SampleLvl1_Exp670_ddMSMS"] = {"file": "./Data/WheatEar/823_SampleLvl1_Exp670_ddMSMS.mzXML"  , "params": "WheatEar"}

    ###############################################
    ## GPU information
    ##
    ## These values specify how the GPU is used for generating the training examples
    ## Please consult the documentation of your GPU.
    ## Values for an old Nvidia GTX 970 graphics card with 4GB GPU-memory are blockdim = 256, griddim = 64
    ## These should thus work for most newer card, however, for maximum performance these should be optimized to the GPU used
    ## The strategy specifies on which device tensorflow shall be executed.
    ## exportBatchSize: specifies how many putative areas shall be exported in one batch
    ## peakBotModelFile: specifies which model to load from the file system
    blockdim = 512
    griddim  = 256
    strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
    exportBatchSize = 2048
    peakBotModelFile = "./temp/PBmodel.model.h5"        
    
    
    ###############################################
    ## Finisehd with specifying LC-HRMS chromatogram files and LC-HRMS settings
    ## Nothing to change from here on
    

    ###############################################
    ### Iterate files and polarities (if FPS is used)
    for inFile, fileProps in inFiles.items():
        tic(label="sample")
    
        ###############################################
        ### data parameters for chromatograms        
        params = expParams[fileProps["params"]]
        
        for polarity, filterLine in params["polarities"].items():
            print("Processing sample '%s', polarity '%s'"%(inFile, polarity))
            walls = []
            backgrounds = []
            errors = []
            tic("instance")
            
            ###############################################
            ### Preprocess chromatogram
            tic("sample")
            mzxml = loadFile(fileProps["file"])
            print(mzxml.getFilterLines())
            mzxml.keepOnlyFilterLine(filterLine)
            print("  | .. filtered mzXML file for %s scan events only"%(polarity))
            mzxml.removeBounds(minRT = params["minRT"], maxRT = params["maxRT"])
            mzxml.removeNoise(params["noiseLevel"])
            print("  | .. removed noise and bounds")
            print("")
            with tempfile.TemporaryDirectory() as tmpdirname:
                
                ###############################################
                ### Detect local maxima with peak-like shapes## CUDA-GPU
                tic(label="preProcessing")
                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
                        mzxml, "'%s':'%s'"%(inFile, filterLine), 
                        intraScanMaxAdjacentSignalDifferencePPM = params["intraScanMaxAdjacentSignalDifferencePPM"],
                        interScanMaxSimilarSignalDifferencePPM = params["interScanMaxSimilarSignalDifferencePPM"],
                        RTpeakWidth = params["RTpeakWidth"],
                        SavitzkyGolayWindowPlusMinus = params["SavitzkyGolayWindowPlusMinus"], 
                        minIntensity = params["minIntensity"],
                        exportPath = tmpdirname, 
                        exportLocalMaxima = "all",
                        exportBatchSize = exportBatchSize, 
                        blockdim = blockdim,
                        griddim  = griddim, 
                        verbose = True)
                print("")
                
                ###############################################
                ### Detect peaks with PeakBot
                tic("PeakBotDetection")
                peaks = []
                with strategy.scope():
                    peaks, walls, backgrounds, errors = peakbot.runPeakBot(tmpdirname, params["PeakBotModel"])
                print("")

            with tempfile.TemporaryDirectory() as tmpdirname:
                ###############################################
                ### Detect high-quality local maxima with peak-like shapes with GD and PeakBot
                peaks, maximaProps, maximaPropsAll = peakbot.cuda.preProcessChromatogram(
                        mzxml, "'%s':'%s'"%(inFile, filterLine), 
                        intraScanMaxAdjacentSignalDifferencePPM = params["intraScanMaxAdjacentSignalDifferencePPM"],
                        interScanMaxSimilarSignalDifferencePPM = params["interScanMaxSimilarSignalDifferencePPM"],
                        RTpeakWidth = params["RTpeakWidth"],
                        SavitzkyGolayWindowPlusMinus = params["SavitzkyGolayWindowPlusMinus"], 
                        minIntensity = params["minIntensity"],
                        exportPath = tmpdirname, 
                        exportLocalMaxima = "peak-like-shape",
                        exportBatchSize = exportBatchSize, 
                        blockdim = blockdim,
                        griddim  = griddim, 
                        verbose = True)
                tic("PeakBotDetection")
                peaks = []
                with strategy.scope():
                    peaks, walls_, backgrounds_, errors_ = peakbot.runPeakBot(tmpdirname, params["PeakBotModel"])
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
                TabLog().addData("%s - %s"%(inFile, filterLine), "Walls", len(walls))
                TabLog().addData("%s - %s"%(inFile, filterLine), "Backgrounds", len(backgrounds))
                TabLog().addData("%s - %s"%(inFile, filterLine), "Errors", len(errors))
                peakbot.exportPeakBotResultsFeatureML(peaks, "%s_%sPeakBot.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity))
                peakbot.exportPeakBotWallsFeatureML(walls, "%s_%sPeakBotWalls.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity))
                peakbot.exportPeakBotWallsFeatureML(backgrounds, "%s_%sPeakBotBackgrounds.featureML"%(fileProps["file"].replace(".mzXML", ""), polarity))
                peakbot.exportMSMSExclusionList(walls, backgrounds, peaks, "%s_%s_MSMSExclusionList.tsv"%(fileProps["file"].replace(".mzXML", ""), polarity))
                print("")
                
                tocP("File '%s':'%s': Exclusion list generated"%(inFile, filterLine), label="instance")
                TabLog().addData("%s - %s"%(inFile, filterLine), "time (sec)", "%.1f"%toc("instance"))
                print("\n\n\n\n\n")
        
        ## Export for an Orbitrap instrument
        headers, eP = peakbot.readTSVFile("%s_%s_MSMSExclusionList.tsv"%(fileProps["file"].replace(".mzXML", ""), "positive") , convertToMinIfPossible = True)
        headers, eN = peakbot.readTSVFile("%s_%s_MSMSExclusionList.tsv"%(fileProps["file"].replace(".mzXML", ""), "negative") , convertToMinIfPossible = True)

        with open("%s_MSMSExclusionList.tsv"%(fileProps["file"].replace(".mzXML", "")), "w") as fout:
            fout.write("Mass [m/z]\tFormula [M]\tFormula type\tSpecies\tCS [z]\tPolarity\tStart [min]\tEnd [min]\t(N)CE\tMSXID\tComment\n")
            for e in eP:
                fout.write("%f\t\t\t\t\t%s\t%f\t%f\t\t\t\n"%(e[0], "positive", float(e[1])/60., float(e[2])/60.))
            for e in eN:
                fout.write("%f\t\t\t\t\t%s\t%f\t%f\t\t\t\n"%(e[0], "negative", float(e[1])/60., float(e[2])/60.))
        
    TabLog().addData("Total time all files", "time (sec)", "%.1f"%toc("overall"))
    print("")
    TabLog().print()
