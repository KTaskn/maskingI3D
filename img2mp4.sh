
# masked00-0.png - masked191-0.png -> masked.mp4
rootdir=./masked
dirs=`ls $rootdir`
for dir in $dirs
do
    echo $rootdir/$dir/09000
    docker run --rm -ti -v $PWD:/work -w /work jrottenberg/ffmpeg -r 30 -i $rootdir/$dir/09000/masked%02d-0.png -vcodec libx264 -pix_fmt yuv420p $rootdir/_videos/${dir}_09000-0.mp4
    docker run --rm -ti -v $PWD:/work -w /work jrottenberg/ffmpeg -r 30 -i $rootdir/$dir/09000/masked%02d-1.png -vcodec libx264 -pix_fmt yuv420p $rootdir/_videos/${dir}_09000-1.mp4
done