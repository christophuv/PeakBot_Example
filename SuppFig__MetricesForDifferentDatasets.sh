

## Run in PeakBot environment

clear


if false; then

    echo ""
    echo ""
    echo ""
    echo "Training new models and verifying their performance"

    python trainPB_WheatEar.py --train --replicates 5
    python trainPB_PHM.py --train --replicates 5
    python trainPB_MTBLS1358.py --train --replicates 5
    python trainPB_MTBLS868.py --train --replicates 5
    python trainPB_MTBLS797.py --train --replicates 5
    
    cd ..
    python FigCompareTrainExamples.py --train --replicates 5
    cd peakbot_example

fi

echo ""
echo ""
echo ""
echo "Generating illustrations"

python trainPB_WheatEar.py 
python trainPB_PHM.py
python trainPB_MTBLS1358.py
python trainPB_MTBLS868.py
python trainPB_MTBLS797.py

cd ..
python FigCompareTrainExamples.py
cd peakbot_example


echo " all done..."
