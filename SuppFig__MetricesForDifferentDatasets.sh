

## Run in PeakBot environment

clear

python trainPB_WheatEar.py --replicates 1


exit

python trainPB_PHM.py --replicates 5

python trainPB_MTBLS1358.py --replicates 5

python trainPB_MTBLS868.py --replicates 5

python trainPB_MTBLS797.py --replicates 5

cd ..

python FigCompareTrainExamples.py

cd peakbot_example

