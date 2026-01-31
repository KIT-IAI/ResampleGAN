import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

def load_and_process_data(csv_file):
    """
    读取并处理CSV数据，分别处理train和test数据
    """
    print("=" * 60)
    print("数据读取和预处理（分离train/test，仅深度学习模型）")
    print("=" * 60)

    # 读取数据
    df = pd.read_csv(csv_file)

    # 重命名列名，处理空列名
    df.columns = ['metric', 'phase', 'structure'] + list(df.columns[3:])

    # 去除第一行（header行）
    df = df.iloc[1:].reset_index(drop=True)

    print(f"原始数据维度: {df.shape}")

    # 过滤掉static相关数据，只保留深度学习模型
    df_filtered = df[df['phase'] != 'static'].reset_index(drop=True)
    # df_filtered = df.copy()

    print(f"过滤static后数据维度: {df_filtered.shape}")
    print(f"保留的Phase: {df_filtered['phase'].unique()}")
    print(f"保留的Structure: {df_filtered['structure'].unique()}")

    # 识别数值列（版本列）
    value_columns = [col for col in df_filtered.columns if col.startswith('v')]
    print(f"\n发现的版本列: {value_columns}")

    # 将宽格式转换为长格式，分别处理train和test
    train_data = []
    test_data = []

    for _, row in df_filtered.iterrows():
        metric = row['metric']
        phase = row['phase']
        structure = row['structure']

        # 处理每个版本的train/test数据
        for i in range(0, len(value_columns), 2):
            if i+1 < len(value_columns):
                version_name = value_columns[i].replace('_1', '').replace('_2', '')
                train_col = value_columns[i]
                test_col = value_columns[i+1] if i+1 < len(value_columns) else None

                # 添加train数据
                if pd.notna(row[train_col]) and row[train_col] != '':
                    train_data.append({
                        'metric': metric,
                        'phase': phase,
                        'structure': structure,
                        'version': version_name,
                        'value': float(row[train_col])
                    })

                # 添加test数据
                if test_col and pd.notna(row[test_col]) and row[test_col] != '':
                    test_data.append({
                        'metric': metric,
                        'phase': phase,
                        'structure': structure,
                        'version': version_name,
                        'value': float(row[test_col])
                    })

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    print(f"\nTrain数据维度: {train_df.shape}")
    print(f"Test数据维度: {test_df.shape}")

    print(f"\nTrain数据分组统计:")
    print(f"Phase数量: {train_df['phase'].nunique()} - {list(train_df['phase'].unique())}")
    print(f"Structure数量: {train_df['structure'].nunique()} - {list(train_df['structure'].unique())}")
    print(f"Metric数量: {train_df['metric'].nunique()} - {list(train_df['metric'].unique())}")

    return train_df, test_df

def welch_anova_oneway(groups):
    """
    执行单因素Welch ANOVA（对方差齐性违反稳健）
    """
    # 计算每组的统计量
    k = len(groups)  # 组数
    ni = np.array([len(group) for group in groups])  # 每组样本量
    xi = np.array([np.mean(group) for group in groups])  # 每组均值
    si2 = np.array([np.var(group, ddof=1) for group in groups])  # 每组方差
    wi = ni / si2  # 权重

    # 计算加权总均值
    x_bar = np.sum(wi * xi) / np.sum(wi)

    # 计算Welch F统计量
    numerator = np.sum(wi * (xi - x_bar)**2) / (k - 1)

    # 计算分母（复杂的权重调整）
    lambda_i = (1 - wi / np.sum(wi))**2 / (ni - 1)
    denominator = 1 + (2 * (k - 2) / (k**2 - 1)) * np.sum(lambda_i)

    welch_f = numerator / denominator

    # 计算自由度
    df1 = k - 1
    df2 = (k**2 - 1) / (3 * np.sum(lambda_i))

    # 计算p值
    p_value = 1 - stats.f.cdf(welch_f, df1, df2)

    return welch_f, p_value, df1, df2

def games_howell_test(data, group_col, value_col):
    """
    执行Games-Howell事后检验（对方差不齐稳健的多重比较）
    """
    groups = data.groupby(group_col)[value_col]
    group_names = list(groups.groups.keys())
    group_data = {name: group.values for name, group in groups}

    # 计算每组的统计量
    group_stats = {}
    for name in group_names:
        values = group_data[name]
        group_stats[name] = {
            'n': len(values),
            'mean': np.mean(values),
            'var': np.var(values, ddof=1),
            'std': np.std(values, ddof=1)
        }

    # 进行两两比较
    results = []

    for group1, group2 in combinations(group_names, 2):
        # 组1和组2的统计量
        n1, mean1, var1 = group_stats[group1]['n'], group_stats[group1]['mean'], group_stats[group1]['var']
        n2, mean2, var2 = group_stats[group2]['n'], group_stats[group2]['mean'], group_stats[group2]['var']

        # 计算t统计量
        mean_diff = mean1 - mean2
        se_diff = np.sqrt(var1/n1 + var2/n2)  # 标准误
        t_stat = mean_diff / se_diff

        # Welch-Satterthwaite自由度
        df = (var1/n1 + var2/n2)**2 / ((var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1))

        # 计算双尾p值
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        # 计算置信区间（95%）
        t_critical = stats.t.ppf(0.975, df)
        ci_lower = mean_diff - t_critical * se_diff
        ci_upper = mean_diff + t_critical * se_diff

        results.append({
            'group1': group1,
            'group2': group2,
            'meandiff': mean_diff,
            'se': se_diff,
            't_stat': t_stat,
            'df': df,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # Bonferroni多重比较校正
    num_comparisons = len(results)
    results_df['p_adj'] = results_df['p_value'].apply(
        lambda p: min(1.0, p * num_comparisons)
    )

    # 判断显著性
    results_df['reject'] = results_df['p_adj'] < 0.05

    return results_df

def perform_welch_anova_analysis(data, metric_name, data_type):
    """
    执行Welch ANOVA分析
    """
    print(f"\n" + "=" * 60)
    print(f"{metric_name} - {data_type}数据的Welch ANOVA分析")
    print("=" * 60)

    # 筛选指定metric的数据
    metric_data = data[data['metric'] == metric_name].copy()

    if len(metric_data) == 0:
        print(f"没有找到 {metric_name} 的数据")
        return None, None

    print(f"分析数据维度: {metric_data.shape}")
    print(f"Phase levels: {metric_data['phase'].unique()}")
    print(f"Structure levels: {metric_data['structure'].unique()}")

    welch_results = {}

    try:
        # Phase的Welch ANOVA
        phase_groups = [group['value'].values for name, group in metric_data.groupby('phase')]
        welch_f_phase, welch_p_phase, df1_phase, df2_phase = welch_anova_oneway(phase_groups)

        print(f"\nPhase效应 (Welch ANOVA):")
        print(f"  F({df1_phase:.0f}, {df2_phase:.2f}) = {welch_f_phase:.4f}, p = {welch_p_phase:.4f}")
        phase_effect = "显著" if welch_p_phase < 0.05 else "不显著"
        print(f"  结果: {phase_effect}")

        # Structure的Welch ANOVA
        structure_groups = [group['value'].values for name, group in metric_data.groupby('structure')]
        welch_f_structure, welch_p_structure, df1_structure, df2_structure = welch_anova_oneway(structure_groups)

        print(f"\nStructure效应 (Welch ANOVA):")
        print(f"  F({df1_structure:.0f}, {df2_structure:.2f}) = {welch_f_structure:.4f}, p = {welch_p_structure:.4f}")
        structure_effect = "显著" if welch_p_structure < 0.05 else "不显著"
        print(f"  结果: {structure_effect}")

        # 保存Welch ANOVA结果
        welch_results = {
            'phase': {
                'F': welch_f_phase, 'p': welch_p_phase,
                'df1': df1_phase, 'df2': df2_phase,
                'significant': welch_p_phase < 0.05
            },
            'structure': {
                'F': welch_f_structure, 'p': welch_p_structure,
                'df1': df1_structure, 'df2': df2_structure,
                'significant': welch_p_structure < 0.05
            }
        }

    except Exception as e:
        print(f"Welch ANOVA计算出错: {e}")
        return None, None

    return metric_data, welch_results

def perform_games_howell_analysis(data, metric_name, data_type, welch_results):
    """
    执行Games-Howell事后检验分析
    """
    print(f"\n" + "=" * 50)
    print(f"{metric_name} - {data_type}数据的Games-Howell事后检验")
    print("=" * 50)

    games_howell_results = {}

    # Phase Games-Howell
    if welch_results['phase']['significant']:
        print(f"\nPhase组间比较 (Games-Howell):")
        try:
            gh_phase = games_howell_test(data, 'phase', 'value')
            print(f"发现 {len(gh_phase)} 个比较")

            # 显示结果
            for _, row in gh_phase.iterrows():
                significance = "显著" if row['reject'] else "不显著"
                print(f"  {row['group1']} vs {row['group2']}: "
                      f"差值={row['meandiff']:.4f}, p_adj={row['p_adj']:.4f}, {significance}")

            games_howell_results['phase'] = gh_phase

        except Exception as e:
            print(f"Phase Games-Howell出错: {e}")
            games_howell_results['phase'] = None
    else:
        print(f"\nPhase效应不显著，跳过Games-Howell检验")
        games_howell_results['phase'] = None

    # Structure Games-Howell
    if welch_results['structure']['significant']:
        print(f"\nStructure组间比较 (Games-Howell):")
        try:
            gh_structure = games_howell_test(data, 'structure', 'value')
            print(f"发现 {len(gh_structure)} 个比较")

            # 显示结果
            for _, row in gh_structure.iterrows():
                significance = "显著" if row['reject'] else "不显著"
                print(f"  {row['group1']} vs {row['group2']}: "
                      f"差值={row['meandiff']:.4f}, p_adj={row['p_adj']:.4f}, {significance}")

            games_howell_results['structure'] = gh_structure

        except Exception as e:
            print(f"Structure Games-Howell出错: {e}")
            games_howell_results['structure'] = None
    else:
        print(f"\nStructure效应不显著，跳过Games-Howell检验")
        games_howell_results['structure'] = None

    return games_howell_results

def save_results_to_dataframes(all_results):
    """
    将所有结果保存到DataFrame中
    """
    print(f"\n" + "=" * 50)
    print("结果汇总到DataFrame")
    print("=" * 50)

    # 1. Welch ANOVA结果表
    welch_anova_results = []

    for data_type in ['train', 'test']:
        for metric in all_results[data_type].keys():
            welch_data = all_results[data_type][metric].get('welch_results')
            if welch_data:
                # Phase结果
                welch_anova_results.append({
                    'data_type': data_type,
                    'metric': metric,
                    'factor': 'phase',
                    'F_statistic': welch_data['phase']['F'],
                    'p_value': welch_data['phase']['p'],
                    'df1': welch_data['phase']['df1'],
                    'df2': welch_data['phase']['df2'],
                    'significant': welch_data['phase']['significant']
                })

                # Structure结果
                welch_anova_results.append({
                    'data_type': data_type,
                    'metric': metric,
                    'factor': 'structure',
                    'F_statistic': welch_data['structure']['F'],
                    'p_value': welch_data['structure']['p'],
                    'df1': welch_data['structure']['df1'],
                    'df2': welch_data['structure']['df2'],
                    'significant': welch_data['structure']['significant']
                })

    welch_anova_df = pd.DataFrame(welch_anova_results)

    # 2. Games-Howell结果表
    games_howell_results = []

    for data_type in ['train', 'test']:
        for metric in all_results[data_type].keys():
            gh_data = all_results[data_type][metric].get('games_howell_results')
            if gh_data:
                # Phase结果
                if gh_data['phase'] is not None:
                    for _, row in gh_data['phase'].iterrows():
                        games_howell_results.append({
                            'data_type': data_type,
                            'metric': metric,
                            'factor': 'phase',
                            'group1': row['group1'],
                            'group2': row['group2'],
                            'meandiff': row['meandiff'],
                            'se': row['se'],
                            't_stat': row['t_stat'],
                            'df': row['df'],
                            'p_value': row['p_value'],
                            'p_adj': row['p_adj'],
                            'significant': row['reject'],
                            'ci_lower': row['ci_lower'],
                            'ci_upper': row['ci_upper']
                        })

                # Structure结果
                if gh_data['structure'] is not None:
                    for _, row in gh_data['structure'].iterrows():
                        games_howell_results.append({
                            'data_type': data_type,
                            'metric': metric,
                            'factor': 'structure',
                            'group1': row['group1'],
                            'group2': row['group2'],
                            'meandiff': row['meandiff'],
                            'se': row['se'],
                            't_stat': row['t_stat'],
                            'df': row['df'],
                            'p_value': row['p_value'],
                            'p_adj': row['p_adj'],
                            'significant': row['reject'],
                            'ci_lower': row['ci_lower'],
                            'ci_upper': row['ci_upper']
                        })

    games_howell_df = pd.DataFrame(games_howell_results)

    # 3. 描述性统计表
    descriptive_results = []

    for data_type in ['train', 'test']:
        for metric in all_results[data_type].keys():
            data = all_results[data_type][metric].get('data')
            if data is not None:
                # 按Phase分组
                phase_stats = data.groupby('phase')['value'].agg(['count', 'mean', 'std', 'min', 'max'])
                for phase, stats in phase_stats.iterrows():
                    descriptive_results.append({
                        'data_type': data_type,
                        'metric': metric,
                        'factor': 'phase',
                        'group': phase,
                        'count': stats['count'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max']
                    })

                # 按Structure分组
                structure_stats = data.groupby('structure')['value'].agg(['count', 'mean', 'std', 'min', 'max'])
                for structure, stats in structure_stats.iterrows():
                    descriptive_results.append({
                        'data_type': data_type,
                        'metric': metric,
                        'factor': 'structure',
                        'group': structure,
                        'count': stats['count'],
                        'mean': stats['mean'],
                        'std': stats['std'],
                        'min': stats['min'],
                        'max': stats['max']
                    })

    descriptive_df = pd.DataFrame(descriptive_results)

    print(f"✅ Welch ANOVA结果表: {welch_anova_df.shape}")
    print(f"✅ Games-Howell结果表: {games_howell_df.shape}")
    print(f"✅ 描述性统计表: {descriptive_df.shape}")

    return welch_anova_df, games_howell_df, descriptive_df

def main_analysis(csv_file):
    """
    主分析函数 - 分离train/test，仅使用稳健方法
    """
    print("深度学习模型稳健显著性检验分析")
    print("（分离train/test，仅使用Welch ANOVA + Games-Howell）")
    print("=" * 60)

    # 1. 数据加载和预处理
    train_df, test_df = load_and_process_data(csv_file)

    # 2. 存储所有结果
    all_results = {'train': {}, 'test': {}}

    # 3. 分别分析train和test数据
    for data_type, data in [('train', train_df), ('test', test_df)]:
        print(f"\n{'='*80}")
        print(f"开始分析 {data_type.upper()} 数据")
        print(f"{'='*80}")

        metrics = data['metric'].unique()
        print(f"发现的评估指标: {metrics}")

        for metric in metrics:
            print(f"\n{'-'*60}")
            print(f"分析 {metric} 指标")
            print(f"{'-'*60}")

            # Welch ANOVA分析
            metric_data, welch_results = perform_welch_anova_analysis(data, metric, data_type)

            if welch_results is not None:
                # Games-Howell事后检验
                games_howell_results = perform_games_howell_analysis(metric_data, metric, data_type, welch_results)

                # 保存结果
                all_results[data_type][metric] = {
                    'data': metric_data,
                    'welch_results': welch_results,
                    'games_howell_results': games_howell_results
                }

    # 4. 将结果保存到DataFrame
    welch_anova_df, games_howell_df, descriptive_df = save_results_to_dataframes(all_results)

    # 5. 显示结果概览
    print(f"\n{'='*80}")
    print("分析结果概览")
    print(f"{'='*80}")

    print(f"\n📊 Welch ANOVA 结果概览:")
    print(welch_anova_df.groupby(['data_type', 'factor'])['significant'].sum())

    print(f"\n📈 Games-Howell 显著比较概览:")
    if len(games_howell_df) > 0:
        print(games_howell_df.groupby(['data_type', 'factor'])['significant'].sum())
    else:
        print("没有需要进行Games-Howell检验的比较")

    return train_df, test_df, all_results, welch_anova_df, games_howell_df, descriptive_df