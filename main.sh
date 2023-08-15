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
    echo $endpoint
    if [ ! -d "./masked/$endpoint" ]; then
        mkdir -p ./masked/$endpoint
    fi
    python main.py --endpoint $endpoint
done