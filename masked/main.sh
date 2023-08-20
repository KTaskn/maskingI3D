# bash
# ENDPOINTS = [
#         'Conv3d_1a_7x7',
#         'MaxPool3d_2a_3x3',
#         'Conv3d_2b_1x1',
#         'Conv3d_2c_3x3',
#         'MaxPool3d_3a_3x3']

for endpoint in 'MaxPool3d_3a_3x3' 'Conv3d_2c_3x3' 'Conv3d_2b_1x1'
# for endpoint in 'Conv3d_2c_3x3'
do
    for alpha in '2.0' '1.0' '0.0'
    do
        for beta in '2.0' '1.0' '0.0'
        do
            for avgpool_kernel_size in '9' '7'
            do
                dirname=${endpoint}_${alpha}_${beta}_${avgpool_kernel_size}
                echo $endpoint
                if [ ! -d "./masked/$dirname" ]; then
                    mkdir -p ./masked/$dirname
                fi

                # 1000 to 10000
                for step in {0..10000..1000}
                do
                    # 0 padding
                    step_dirname=`printf "%05d" "${step}"`
                    if [ ! -d "./masked/$dirname/$step_dirname" ]; then
                        mkdir -p ./masked/$dirname/$step_dirname
                    fi
                done

                python main.py --endpoint $endpoint --alpha $alpha --beta $beta --avgpool_kernel_size $avgpool_kernel_size
            done
        done
    done
done