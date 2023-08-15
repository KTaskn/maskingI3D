# bash
# ENDPOINTS = [
#         'Conv3d_1a_7x7',
#         'MaxPool3d_2a_3x3',
#         'Conv3d_2b_1x1',
#         'Conv3d_2c_3x3',
#         'MaxPool3d_3a_3x3']

for endpoint in 'Conv3d_1a_7x7' 'MaxPool3d_2a_3x3' 'Conv3d_2b_1x1' 'Conv3d_2c_3x3' 'MaxPool3d_3a_3x3' \
             'Mixed_3b' 'Mixed_3c' 'MaxPool3d_4a_3x3' 'Mixed_4b' 'Mixed_4c' 'Mixed_4d' 'Mixed_4e' 'Mixed_4f' 'MaxPool3d_5a_2x2'
do
    python main.py --endpoint $endpoint
    # if [ ! -d "./masked/$endpoint" ]; then
    #     mkdir -p ./masked/$endpoint
    # fi
    # python main.py --endpoint $endpoint &> ./masked/$endpoint/log.txt
done