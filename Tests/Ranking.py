import json
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt


def sorting_values(arr, reverse=True):
    sorted_arr = sorted(set(arr), reverse=reverse)
    position_dict = {value: index for index, value in enumerate(sorted_arr)}
    sorting_values = [position_dict[element] + 1 for element in arr]
    return np.array(sorting_values)


if __name__ == '__main__':
    model_names = ['3DUNet', 'STUNet', 'AttentionUNet', 'SwinUNETR', 'FocalSegNet', '3DSwinUnet', '3DSwinUnetV4']

    # Conventional
    inference_rate = np.array([6.158, 7.298, 7.193, 18.585, 15.412, 22.910, 29.025])

    # 4 Stages
    # inference_rate = np.array([5.108, 6.052, 5.970, 18.585, 15.412, 28.611])

    # Conventional
    param_counts = np.array([30.6, 30.23, 30.59, 62.19, 69.65, 7.98, 31.55])

    # 4 Stages
    # param_counts = np.array([14.59, 14.51, 14.77, 62.19, 69.65, 30.91])

    with open(f'Results/all_models_accuracy.json', 'r') as json_file:
        all_models_accuracy = json_file.read()
    all_models_accuracy = json.loads(all_models_accuracy)
    conventional_model = ['3DUNet', 'STUNet', 'AttentionUNet', 'SUR', 'FocalSegNet', 'SwinUnet', 'SwinUNetV4']
    four_staged_model = ['3DUNetS4', 'STUNetS5', 'AttentionUNetS4', 'SUR', 'FocalSegNet', 'SwinUNetS4']

    dice_btcv = np.array([all_models_accuracy[k]['BTCVDice'][0] for k in conventional_model])
    dice_btcv_std = np.array([all_models_accuracy[k]['BTCVDice'][1] for k in conventional_model])
    dice_surg = np.array([all_models_accuracy[k]['SurgDice'][0] for k in conventional_model])
    dice_sur_std = np.array([all_models_accuracy[k]['SurgDice'][1] for k in conventional_model])
    dice_total = np.array([all_models_accuracy[k]['AllDice'][0] for k in conventional_model])
    dice_total_std = np.array([all_models_accuracy[k]['AllDice'][1] for k in conventional_model])

    nsd_btcv = np.array([all_models_accuracy[k]['BTCVNSD'][0] for k in conventional_model])
    nsd_btcv_std = np.array([all_models_accuracy[k]['BTCVNSD'][1] for k in conventional_model])
    nsd_surg = np.array([all_models_accuracy[k]['SurgNSD'][0] for k in conventional_model])
    nsd_surg_std = np.array([all_models_accuracy[k]['SurgNSD'][1] for k in conventional_model])
    nsd_total = np.array([all_models_accuracy[k]['AllNSD'][0] for k in conventional_model])
    nsd_total_std = np.array([all_models_accuracy[k]['AllNSD'][1] for k in conventional_model])

    # 4 Stages
    # dice_btcv = np.array([all_models_accuracy[k]['BTCVDice'][0] for k in four_staged_model])
    # dice_btcv_std = np.array([all_models_accuracy[k]['BTCVDice'][1] for k in four_staged_model])
    # dice_surg = np.array([all_models_accuracy[k]['SurgDice'][0] for k in four_staged_model])
    # dice_sur_std = np.array([all_models_accuracy[k]['SurgDice'][1] for k in four_staged_model])
    # dice_total = np.array([all_models_accuracy[k]['AllDice'][0] for k in four_staged_model])
    # dice_total_std = np.array([all_models_accuracy[k]['AllDice'][1] for k in four_staged_model])
    #
    # nsd_btcv = np.array([all_models_accuracy[k]['BTCVNSD'][0] for k in four_staged_model])
    # nsd_btcv_std = np.array([all_models_accuracy[k]['BTCVNSD'][1] for k in four_staged_model])
    # nsd_surg = np.array([all_models_accuracy[k]['SurgNSD'][0] for k in four_staged_model])
    # nsd_surg_std = np.array([all_models_accuracy[k]['SurgNSD'][1] for k in four_staged_model])
    # nsd_total = np.array([all_models_accuracy[k]['AllNSD'][0] for k in four_staged_model])
    # nsd_total_std = np.array([all_models_accuracy[k]['AllNSD'][1] for k in four_staged_model])

    param_count_ranks = sorting_values(param_counts, reverse=False)
    inference_rate_ranks = sorting_values(inference_rate, reverse=False)
    print(param_count_ranks)

    complexity_rank = sorting_values(param_count_ranks + inference_rate_ranks, reverse=False)
    dice_btcv_ranks = sorting_values(dice_btcv)
    dice_surg_ranks = sorting_values(dice_surg)
    dice_total_ranks = sorting_values(dice_total)
    nsd_btcv_ranks = sorting_values(nsd_btcv)
    nsd_surg_ranks = sorting_values(nsd_surg)
    nsd_total_ranks = sorting_values(nsd_total)

    segmentation_rank_btcv = sorting_values(dice_btcv_ranks + nsd_btcv_ranks, reverse=False)
    segmentation_rank_surg = sorting_values(dice_surg_ranks + nsd_surg_ranks, reverse=False)
    segmentation_rank_total = sorting_values(dice_total_ranks + nsd_total_ranks, reverse=False)

    print(segmentation_rank_btcv)
    print(segmentation_rank_surg)
    print(segmentation_rank_total)

    btcv_final_rank = sorting_values(dice_btcv_ranks + nsd_btcv_ranks + param_count_ranks + inference_rate_ranks,
                                     reverse=False)
    surg_final_rank = sorting_values(dice_surg_ranks + nsd_surg_ranks + param_count_ranks + inference_rate_ranks,
                                     reverse=False)
    total_final_rank = sorting_values(dice_total_ranks + nsd_total_ranks + param_count_ranks + inference_rate_ranks,
                                      reverse=False)

    for m in range(len(model_names)):
        print()
        print(f'${model_names[m]}$ '
              f'& ${param_counts[m]} ^{param_count_ranks[m]}$ & ${inference_rate[m]} ^{inference_rate_ranks[m]}$ &'
              f' {complexity_rank[m]}'
              f' & ${dice_btcv[m]:.2f} \pm {dice_btcv_std[m]:.2f} ^{dice_btcv_ranks[m]}$ & ${dice_surg[m]:.2f} '
              f'\pm {dice_sur_std[m]:.2f} ^{dice_surg_ranks[m]}$ & ${dice_total[m]:.2f} \pm {dice_total_std[m]:.2f} ^{dice_total_ranks[m]}$'
              f' & ${nsd_btcv[m]:.2f} \pm {nsd_btcv_std[m]:.2f} ^{nsd_btcv_ranks[m]}$ & ${nsd_surg[m]:.2f}'
              f' \pm {nsd_surg_std[m]:.2f} ^{nsd_surg_ranks[m]}$ & ${nsd_total[m]:.2f} \pm {nsd_total_std[m]:.2f} ^{nsd_total_ranks[m]}$'
              f' & {segmentation_rank_btcv[m]} & {segmentation_rank_surg[m]} & {segmentation_rank_total[m]}'
              f' & {btcv_final_rank[m]} & {surg_final_rank[m]} & {total_final_rank[m]} \\\\'
              )

        # print(f'${model_names[m]}$ & {param_count_ranks[m]} & {inference_rate_ranks[m]} &'
        #       f' {complexity_rank[m]}'
        #       f' & {dice_btcv_ranks[m]} & {dice_surg_ranks[m]} & {dice_total_ranks[m]}'
        #       f' & {nsd_btcv_ranks[m]} & {nsd_surg_ranks[m]} & {nsd_total_ranks[m]}'
        #       f' & {segmentation_rank_btcv[m]} & {segmentation_rank_surg[m]} & {segmentation_rank_total[m]}'
        #       f' & {btcv_final_rank[m]} & {surg_final_rank[m]} & {total_final_rank[m]} \\\\'
        #       )
        print('\\hline')
