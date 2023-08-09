dir=/datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test
outdir=./UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test

for subdir in $dir/*
do
    subdir=`basename $subdir`
    # if directory does not exist, create it
    # mkdir ${outdir}/${subdir}
    # mkdir ${outdir}/${subdir}/flows
    # mkdir ${outdir}/${subdir}/images
    if [ ! -d "${outdir}/${subdir}/flows" ]; then
        mkdir -p ${outdir}/${subdir}/flows
    fi
    if [ ! -d "${outdir}/${subdir}/images" ]; then
        mkdir -p ${outdir}/${subdir}/images
    fi
    python tvl1.py --dir /datasets/UCSD_Anomaly_Dataset_v1p2/UCSDped1/Test/${subdir} --outdir ${outdir}/${subdir}
done
