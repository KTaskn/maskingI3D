# bash

# Test001 to Test036
for i in {1..36}
do
    casename=`printf "Test%03d" $i`
    echo $casename
    output='masked_eachcase'
    endpoint='MaxPool3d_3a_3x3'
    alpha='0.0'
    beta='0.0'
    avgpool_kernel_size='9'
    dirname=${casename}_${endpoint}_${alpha}_${beta}_${avgpool_kernel_size}
    if [ ! -d "./$output/$dirname" ]; then
        mkdir -p ./$output/$dirname
    fi

    # 1000 to 10000
    for step in {0..10000..1000}
    do
        # 0 padding
        step_dirname=`printf "%05d" "${step}"`
        if [ ! -d "./$output/$dirname/$step_dirname" ]; then
            mkdir -p ./$output/$dirname/$step_dirname
        fi
    done

    python main.py --output $output --endpoint $endpoint --alpha $alpha --beta $beta --avgpool_kernel_size $avgpool_kernel_size --casename $casename
done