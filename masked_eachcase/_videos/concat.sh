ls ./stacked/*.mp4 | sed 's/^/file /' > _mylist.txt

# for i in {1..36}
# do
#     casename=`printf "Test%03d" $i`
#     filename=${casename}'_MaxPool3d_3a_3x3_0.0_0.0_9_10000'
#     docker run --rm -ti -v $PWD:/work -w /work jrottenberg/ffmpeg -i ./unstack/${filename}-0.mp4 -i ./unstack/${filename}-1.mp4 -filter_complex "hstack" ./stacked/${casename}.mp4
# done
docker run --rm -ti -v $PWD:/work -w /work jrottenberg/ffmpeg -safe 0 -f concat -i _mylist.txt -c copy output.mp4