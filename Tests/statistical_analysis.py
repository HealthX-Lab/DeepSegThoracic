import os
import re
import json
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import pandas as pd


def perform_ttest(model1, model2):
    # Perform the independent samples t-test
    # print(model1)
    # print(model2)
    t_statistic, p_value = stats.ttest_ind(model1, model2)

    # Print the results
    print(f"t-statistic: {t_statistic}")
    print(f"p-value: {p_value}")

    # Determine the significance
    alpha = 0.05  # Set your significance level (e.g., 0.05)
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
        return True
    else:
        print("Fail to reject the null hypothesis: There is no significant difference between the groups.")
        return False


if __name__ == '__main__':

    root = 'results\\NSD_Dice_full'
    titles = ['AttentionUNet', 'AttentionUNetS4', 'nnunet', 'nnunetS4', 'nnunet_slidingWindow', 'nnunet_slidingWindowS4', 'STUNet', 'STUNetS5', 'SwinUNETR',
              'SwinUNETRV2', 'SwinUNetP2', 'SwinUNetS4', 'SwinUNetV1', 'SwinUNetV2',
              'SwinUNetV3', 'SwinUNetV3P2', 'SwinUNetV4']
    all_pred_folders = os.listdir(root)
    all_pred_folders.remove('nnunet_test_info_predictedTs_AU_ModifiedPoly.json')
    all_pred_folders.remove('nnunet_test_info_predictedTs.json')
    # all_pred_folders.remove('complete_stats')
    # all_pred_folders.remove('old_11_8_2023')
    # all_pred_folders.remove('temp')
    # print(all_pred_folders)
    model_results = {}
    for f in all_pred_folders:
        with open(f'{root}/{f}', 'r') as json_file:
            test_metrics = json_file.read()
        test_metrics = json.loads(test_metrics)

        # model_name = re.findall(r'nnunet_test_results_(.+)\.json', f)[0]
        model_name = re.findall(r'nnunet_test_info_predictedTs_(.+)\.json', f)[0]
        model_results[model_name] = {
            'BTCVDice': test_metrics['BTCVDice'],
            'BTCVNSD': test_metrics['BTCVNSD'],
            'SurgDice': test_metrics['SurgDice'],
            'SurgNSD': test_metrics['SurgNSD'],
            'AllDice': test_metrics['AllDice'],
            'AllNSD': test_metrics['AllNSD']
        }
    print(model_results)

    # Analysis for num of stages
    comparisons = [
        # ('nnunet', 'nnunetS4')
                   ('nnunet_slidingWindow', 'nnunet_slidingWindowS4'), ('AttentionUnet', 'AttentionUnetS4'), ('STUNet', 'STUNetS5'),
                   ('SwinUnetP2', 'SwinUnetS4')]

    rejected_null_hyp = []
    for c in comparisons:
        m_name1, m_name2 = c
        print(f'{m_name1} vs {m_name2}:')
        for k in model_results[m_name1].keys():
            print(k)
            rj = perform_ttest(model_results[m_name1][k], model_results[m_name2][k])
            if rj:
                rejected_null_hyp.append((m_name1, m_name2, k))
            print(50 * '-')
        print(100 * '-')

    for r in rejected_null_hyp:
        print(r)

    print('Anova between Stages:')
    data = {'Model': [], 'Class': [], 'Dice': [], 'NSD': []}
    class_info = [('BTCV', 'BTCVDice', 'BTCVNSD'), ('surgery', 'SurgDice', 'SurgNSD'), ('All', 'AllDice', 'AllNSD')]
    # selected_models = ['nnunetS4', 'STUNetS5', 'AttentionUnetS4', 'SUR', 'SwinUnetS4']
    selected_models = ['nnunet_slidingWindowS4', 'STUNetS5', 'AttentionUnetS4', 'SUR', 'SwinUnetS4']
    # for model_name in model_results.keys():
    for model_name in selected_models:
        for i in range(len(model_results[model_name]['AllDice'])):
            for c in class_info:
                data['Model'].append(model_name)
                data['Class'].append(c[0])
                data['Dice'].append(model_results[model_name][c[1]][i])
                data['NSD'].append(model_results[model_name][c[2]][i])

    print(data)

    df = pd.DataFrame(data)

    print(df)

    # Fit a two-way ANOVA model with interactions
    model = ols('NSD ~ Model * C(Class)', data=df).fit()
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(anova_table)

    model = ols('Dice ~ Model * C(Class)', data=df).fit()
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(anova_table)

    # Perform Tukey's HSD post hoc test
    for c in class_info:
        print(c[0], 'Dice')
        selected_df = df[df['Class'] == c[0]]
        model = ols('Dice ~ Model * C(Class)', data=selected_df).fit()
        posthoc = pairwise_tukeyhsd(selected_df['Dice'], selected_df['Model'])
        print("\nTukey's HSD Post Hoc Test:")
        print(posthoc)
        print(20 * '-')
        print(c[0], 'NSD')
        selected_df = df[df['Class'] == c[0]]
        model = ols('NSD ~ Model * C(Class)', data=selected_df).fit()
        posthoc = pairwise_tukeyhsd(selected_df['NSD'], selected_df['Model'])
        print("\nTukey's HSD Post Hoc Test:")
        print(posthoc)
        print(100 * '---')

    # # Perform Tukey's HSD post hoc test within each class
    # unique_classes = df_class['Class'].unique()
    # for class_name in unique_classes:
    #     class_data = df_class[df_class['Class'] == class_name]
    #     # print(class_data)
    #     posthoc = pairwise_tukeyhsd(class_data['MeanDice'], class_data['Model'])
    #     print(f"\nTukey's HSD Post Hoc Test for {class_name}:")
    #     print(posthoc)
    #     print(100 * '-')

    print('Next Analysis')
    # Analysis for SwinUnet
    comparisons = [('SwinUnet', 'SwinUnetV1'), ('SwinUnetV3', 'SwinUnetV4')]

    rejected_null_hyp = []
    for c in comparisons:
        m_name1, m_name2 = c
        print(f'{m_name1} vs {m_name2}:')
        for k in model_results[m_name1].keys():
            print(k)
            rj = perform_ttest(model_results[m_name1][k], model_results[m_name2][k])
            if rj:
                rejected_null_hyp.append((m_name1, m_name2, k))
            print(50 * '-')
        print(100 * '-')

    for r in rejected_null_hyp:
        print(r)

    print('Anova between swinUnet with different upsampling modules:')
    data = {'Model': [], 'Class': [], 'Dice': [], 'NSD': []}
    class_info = [('BTCV', 'BTCVDice', 'BTCVNSD'), ('surgery', 'SurgDice', 'SurgNSD'), ('All', 'AllDice', 'AllNSD')]
    selected_models = ['SwinUnetV1', 'SwinUnetV2', 'SwinUnetV3']
    # for model_name in model_results.keys():
    for model_name in selected_models:
        for i in range(len(model_results[model_name]['AllDice'])):
            for c in class_info:
                data['Model'].append(model_name)
                data['Class'].append(c[0])
                data['Dice'].append(model_results[model_name][c[1]][i])
                data['NSD'].append(model_results[model_name][c[2]][i])

    print(data)

    df = pd.DataFrame(data)

    print(df)

    # Fit a two-way ANOVA model with interactions
    model = ols('NSD ~ Model * C(Class)', data=df).fit()
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(anova_table)

    model = ols('Dice ~ Model * C(Class)', data=df).fit()
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(anova_table)

    # Perform Tukey's HSD post hoc test
    for c in class_info:
        print(c[0], 'Dice')
        selected_df = df[df['Class'] == c[0]]
        model = ols('Dice ~ Model * C(Class)', data=selected_df).fit()
        posthoc = pairwise_tukeyhsd(selected_df['Dice'], selected_df['Model'])
        print("\nTukey's HSD Post Hoc Test:")
        print(posthoc)
        print(20 * '-')
        print(c[0], 'NSD')
        selected_df = df[df['Class'] == c[0]]
        model = ols('NSD ~ Model * C(Class)', data=selected_df).fit()
        posthoc = pairwise_tukeyhsd(selected_df['NSD'], selected_df['Model'])
        print("\nTukey's HSD Post Hoc Test:")
        print(posthoc)
        print(100 * '---')

    print('Next Analysis')
    # Analysis for original models

    print('Anova between conventional models:')
    data = {'Model': [], 'Class': [], 'Dice': [], 'NSD': []}
    class_info = [('BTCV', 'BTCVDice', 'BTCVNSD'), ('surgery', 'SurgDice', 'SurgNSD'), ('All', 'AllDice', 'AllNSD')]
    # selected_models = ['nnunet', 'STUNet', 'AttentionUnet', 'SUR', 'SwinUnet']
    selected_models = ['nnunet_slidingWindow', 'STUNet', 'AttentionUnet', 'SUR', 'SwinUnet']
    # for model_name in model_results.keys():
    for model_name in selected_models:
        for i in range(len(model_results[model_name]['AllDice'])):
            for c in class_info:
                data['Model'].append(model_name)
                data['Class'].append(c[0])
                data['Dice'].append(model_results[model_name][c[1]][i])
                data['NSD'].append(model_results[model_name][c[2]][i])

    print(data)

    df = pd.DataFrame(data)

    print(df)

    # Fit a two-way ANOVA model with interactions
    model = ols('NSD ~ Model * C(Class)', data=df).fit()
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(anova_table)

    model = ols('Dice ~ Model * C(Class)', data=df).fit()
    # Perform the ANOVA
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Print the ANOVA table
    print(anova_table)

    # Perform Tukey's HSD post hoc test
    for c in class_info:
        print(c[0], 'Dice')
        selected_df = df[df['Class'] == c[0]]
        model = ols('Dice ~ Model * C(Class)', data=selected_df).fit()
        posthoc = pairwise_tukeyhsd(selected_df['Dice'], selected_df['Model'])
        print("\nTukey's HSD Post Hoc Test:")
        print(posthoc)
        print(20 * '-')
        print(c[0], 'NSD')
        selected_df = df[df['Class'] == c[0]]
        model = ols('NSD ~ Model * C(Class)', data=selected_df).fit()
        posthoc = pairwise_tukeyhsd(selected_df['NSD'], selected_df['Model'])
        print("\nTukey's HSD Post Hoc Test:")
        print(posthoc)
        print(100 * '---')
