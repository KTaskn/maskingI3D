# bash
# ENDPOINTS = [
#         'Conv3d_1a_7x7',
#         'MaxPool3d_2a_3x3',
#         'Conv3d_2b_1x1',
#         'Conv3d_2c_3x3',
#         'MaxPool3d_3a_3x3']

# for endpoint in 'MaxPool3d_3a_3x3' 'Conv3d_2c_3x3' 'Conv3d_2b_1x1' 'MaxPool3d_2a_3x3' 'Conv3d_1a_7x7'
for endpoint in 'Conv3d_2c_3x3'
do
    for alpha in '100.0' '50.0' '25.0' '10.0' '5.0' '2.0' '1.0' '0.5' '0.2' '0.1'
    do
        dirname=${endpoint}_${alpha}
        echo $endpoint
        if [ ! -d "./masked/$dirname" ]; then
            mkdir -p ./masked/$dirname
        fi
        python main.py --endpoint $endpoint --alpha $alpha
    done
done