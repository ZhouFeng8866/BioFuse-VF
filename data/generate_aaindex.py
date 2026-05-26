#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: qyguo
"""
AAindex 特征生成脚本
基于 AAindex1 数据库生成蛋白质序列的理化性质特征用于毒力因子多分类任务

AAindex1 包含566种氨基酸理化性质指数，每种指数为20种标准氨基酸提供数值

使用方法:
    python generate_aaindex.py --dataset train
    python generate_aaindex.py --dataset valid
    python generate_aaindex.py --dataset test
    python generate_aaindex.py --dataset all
"""

# ============================================================================
# 公共库导入
# ============================================================================
import os
import sys
import argparse
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import torch

# 设置随机种子
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置所有随机种子以保证可复现性"""
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_label_from_id(seq_id):
    """
    从序列ID中提取标签索引

    参数:
        seq_id: 序列ID，格式为 VFC0XXX-数字

    返回:
        label: 标签索引 (0-6)
    """
    label_dict = {
        'VFC0272': 0,  # Nutritional/Metabolic factor
        'VFC0001': 1,  # Adherence
        'VFC0086': 2,  # Effector delivery system
        'VFC0204': 3,  # Motility
        'VFC0235': 4,  # Exotoxin
        'VFC0258': 5,  # Immune modulation
        'VFC0271': 6   # Biofilm
    }

    # 提取前7个字符作为类别标识
    vfc_id = seq_id[:7]
    return label_dict.get(vfc_id, -1)


def parse_aaindex1(aaindex_file):
    """
    解析 AAindex1 数据库文件

    AAindex1 文件格式:
        H XXXXX - 条目标识符
        D ... - 描述
        R ... - 参考文献
        A ... - 作者
        T ... - 标题
        J ... - 期刊
        C ... - 相关性
        I A/L R/K ... - 氨基酸顺序（第一行）
          数值1 数值2 ... - 第一行数值（10个）
          数值11 数值12 ... - 第二行数值（10个）
        // - 条目结束

    参数:
        aaindex_file: AAindex1 文件路径

    返回:
        aaindex_dict: 字典，键为条目ID，值为20种氨基酸的数值字典
    """
    print(f"\n{'='*60}")
    print(f"正在解析 AAindex1 数据库: {aaindex_file}")
    print(f"{'='*60}")

    aaindex_dict = {}
    current_id = None
    current_values = {}
    aa_order = []

    with open(aaindex_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # 新条目开始
        if line.startswith('H '):
            current_id = line[2:].strip()
            current_values = {}
            aa_order = []

        # 氨基酸数值行
        elif line.startswith('I '):
            # 第一行：氨基酸字母（A/L R/K N/M ...）
            aa_line = line[2:].strip().split()
            aa_order = []
            for pair in aa_line:
                if '/' in pair:
                    aa1, aa2 = pair.split('/')
                    aa_order.extend([aa1, aa2])

            # 第二行和第三行：对应的数值（共20个）
            values = []

            # 读取第二行（前10个数值）
            i += 1
            if i < len(lines):
                value_line1 = lines[i].strip().split()
                for val in value_line1:
                    try:
                        values.append(float(val))
                    except ValueError:
                        # NA 或其他非数值，使用 NaN
                        values.append(np.nan)

            # 读取第三行（后10个数值）
            i += 1
            if i < len(lines):
                value_line2 = lines[i].strip().split()
                for val in value_line2:
                    try:
                        values.append(float(val))
                    except ValueError:
                        # NA 或其他非数值，使用 NaN
                        values.append(np.nan)

            # 构建氨基酸到数值的映射
            if len(aa_order) == 20 and len(values) == 20:
                for aa, val in zip(aa_order, values):
                    current_values[aa] = val

        # 条目结束
        elif line == '//':
            if current_id and len(current_values) == 20:
                aaindex_dict[current_id] = current_values
            current_id = None
            current_values = {}
            aa_order = []

        i += 1

    print(f"✓ 成功解析 {len(aaindex_dict)} 个 AAindex 条目")

    # 检查是否有缺失值
    entries_with_na = 0
    for entry_id, values in aaindex_dict.items():
        if any(np.isnan(v) for v in values.values()):
            entries_with_na += 1

    if entries_with_na > 0:
        print(f"  警告: {entries_with_na} 个条目包含缺失值(NA)")

    return aaindex_dict


def create_extended_aaindex_mapping(aaindex_dict):
    """
    为每个AAindex条目创建扩展的氨基酸映射（包含特殊字符）

    特殊字符处理规则:
        B (Asx): (N + D) / 2
        Z (Glx): (Q + E) / 2
        J (Xle): (L + I) / 2
        O (Pyl): K
        U (Sec): C
        X (Unknown): 0
        - (Gap): 删除

    参数:
        aaindex_dict: 标准20种氨基酸的AAindex字典

    返回:
        extended_dict: 扩展后的字典，每个条目包含所有可能的氨基酸
    """
    extended_dict = {}

    for entry_id, aa_values in aaindex_dict.items():
        extended_values = aa_values.copy()

        # B = (N + D) / 2
        if 'N' in aa_values and 'D' in aa_values:
            extended_values['B'] = (aa_values['N'] + aa_values['D']) / 2
        else:
            extended_values['B'] = 0.0

        # Z = (Q + E) / 2
        if 'Q' in aa_values and 'E' in aa_values:
            extended_values['Z'] = (aa_values['Q'] + aa_values['E']) / 2
        else:
            extended_values['Z'] = 0.0

        # J = (L + I) / 2
        if 'L' in aa_values and 'I' in aa_values:
            extended_values['J'] = (aa_values['L'] + aa_values['I']) / 2
        else:
            extended_values['J'] = 0.0

        # U = C (硒代半胱氨酸类似半胱氨酸)
        if 'C' in aa_values:
            extended_values['U'] = aa_values['C']
        else:
            extended_values['U'] = 0.0

        # O = K (吡咯赖氨酸类似赖氨酸)
        if 'K' in aa_values:
            extended_values['O'] = aa_values['K']
        else:
            extended_values['O'] = 0.0

        # X = 0 (未知氨基酸使用零向量)
        extended_values['X'] = 0.0

        # * (终止密码子) = 0
        extended_values['*'] = 0.0

        extended_dict[entry_id] = extended_values

    return extended_dict


def encode_sequence_with_aaindex(sequence, extended_aaindex_dict):
    """
    使用AAindex编码单个蛋白质序列

    对序列中的每个氨基酸，提取所有AAindex条目的数值，
    然后对整个序列取平均，得到序列级别的特征向量

    参数:
        sequence: 蛋白质序列字符串
        extended_aaindex_dict: 扩展的AAindex字典

    返回:
        feature_vector: numpy数组，形状为 (num_aaindex_entries,)
    """
    # 去除gap字符
    sequence = sequence.replace('-', '').upper()

    if len(sequence) == 0:
        # 空序列返回零向量
        num_entries = len(extended_aaindex_dict)
        return np.zeros(num_entries)

    # 获取所有AAindex条目ID（保持顺序一致）
    entry_ids = sorted(extended_aaindex_dict.keys())

    # 对每个位置的氨基酸，提取所有AAindex特征
    sequence_features = []

    for aa in sequence:
        aa_features = []
        for entry_id in entry_ids:
            aa_values = extended_aaindex_dict[entry_id]
            # 如果氨基酸不在映射中，使用0
            value = aa_values.get(aa, 0.0)
            # 处理NaN值
            if np.isnan(value):
                value = 0.0
            aa_features.append(value)
        sequence_features.append(aa_features)

    # 对序列所有位置取平均，得到序列级别的特征
    sequence_features = np.array(sequence_features)
    feature_vector = np.mean(sequence_features, axis=0)

    return feature_vector


def extract_aaindex_features_from_fasta(fasta_file, aaindex_file):
    """
    从FASTA文件中提取AAindex特征

    参数:
        fasta_file: FASTA文件路径
        aaindex_file: AAindex1文件路径

    返回:
        features_with_labels: torch.Tensor，形状为 (num_sequences, num_features+1)
                             第一列为标签，其余列为AAindex特征
    """
    print(f"\n{'='*60}")
    print(f"正在从 FASTA 文件提取 AAindex 特征")
    print(f"FASTA 文件: {fasta_file}")
    print(f"{'='*60}")

    # 解析AAindex数据库
    aaindex_dict = parse_aaindex1(aaindex_file)

    # 创建扩展的氨基酸映射
    extended_aaindex_dict = create_extended_aaindex_mapping(aaindex_dict)

    print(f"\n正在编码蛋白质序列...")

    # 读取FASTA文件
    records = list(SeqIO.parse(fasta_file, "fasta"))

    features = []
    labels = []

    replacement_stats = {}  # 统计特殊字符

    for record in tqdm(records, desc="AAindex 特征提取"):
        # 提取标签
        label = get_label_from_id(record.id)
        if label == -1:
            print(f"\n警告: 无法识别序列ID {record.id} 的标签，跳过该序列")
            continue

        # 处理序列
        seq = str(record.seq).upper()

        # 统计特殊字符
        for ch in seq:
            if ch not in 'ACDEFGHIKLMNPQRSTVWY':
                replacement_stats[ch] = replacement_stats.get(ch, 0) + 1

        # 编码序列
        feature_vector = encode_sequence_with_aaindex(seq, extended_aaindex_dict)

        features.append(feature_vector)
        labels.append(label)

    # 显示特殊字符统计
    if replacement_stats:
        print("\n特殊字符统计:")
        for ch, count in sorted(replacement_stats.items(), key=lambda x: -x[1]):
            if ch == '-':
                print(f"  '{ch}' (Gap): {count} (已删除)")
            elif ch == '*':
                print(f"  '{ch}' (Stop): {count} (编码为0)")
            elif ch in 'BZJUO':
                print(f"  '{ch}': {count} (已处理)")
            elif ch == 'X':
                print(f"  '{ch}' (Unknown): {count} (编码为0)")
            else:
                print(f"  '{ch}': {count}")

    # 转换为tensor
    features_array = np.array(features)
    labels_array = np.array(labels).reshape(-1, 1)

    # 拼接标签和特征
    features_with_labels = np.concatenate([labels_array, features_array], axis=1)
    features_with_labels = torch.from_numpy(features_with_labels).float()

    num_features = features_array.shape[1]

    print(f"\n✓ AAindex 特征提取完成，形状: {features_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-{num_features} 列: AAindex 特征 ({num_features}维)")
    print(f"  - 使用了 {len(extended_aaindex_dict)} 个 AAindex 条目")

    return features_with_labels


def main():
    """
    主函数：处理命令行参数并生成AAindex特征
    """
    parser = argparse.ArgumentParser(description='为 PLM-GNN 生成 AAindex 特征')
    parser.add_argument('--dataset', type=str,
                       choices=['train', 'valid', 'test', 'all', 'new_test'],
                       default='all',
                       help='选择要处理的数据集')
    parser.add_argument('--fasta_dir', type=str,
                       default='./fasta',
                       help='包含 FASTA 文件的目录')
    parser.add_argument('--aaindex_file', type=str,
                       default='./aaindex/aaindex1.txt',
                       help='AAindex1 数据库文件路径')
    parser.add_argument('--output_dir', type=str,
                       default='./aaindex',
                       help='保存特征的目录')

    args = parser.parse_args()

    # 设置随机种子
    set_seed()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 检查AAindex文件是否存在
    if not os.path.exists(args.aaindex_file):
        print(f"错误: AAindex 文件不存在: {args.aaindex_file}")
        sys.exit(1)

    # 确定要处理的数据集
    if args.dataset == 'all':
        datasets = ['train', 'valid', 'test']
    else:
        datasets = [args.dataset]

    # 文件名映射
    file_mapping = {
        'train': 'train_set.fasta',
        'valid': 'valid_set.fasta',
        'test': 'test_set.fasta',
        'new_test': 'new_test_set.fasta',
    }

    print(f"\n{'='*60}")
    print(f"AAindex 特征生成")
    print(f"{'='*60}")
    print(f"数据集: {', '.join(datasets)}")
    print(f"FASTA 目录: {args.fasta_dir}")
    print(f"AAindex 文件: {args.aaindex_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    # 处理每个数据集
    for dataset in datasets:
        fasta_file = os.path.join(args.fasta_dir, file_mapping[dataset])

        if not os.path.exists(fasta_file):
            print(f"警告: 文件 {fasta_file} 不存在，跳过")
            continue

        print(f"\n{'#'*60}")
        print(f"正在处理 {dataset} 数据集: {fasta_file}")
        print(f"{'#'*60}")

        try:
            # 提取AAindex特征
            features = extract_aaindex_features_from_fasta(
                fasta_file,
                args.aaindex_file
            )

            # 保存特征
            output_file = os.path.join(args.output_dir, f'{dataset}_aaindex.pt')
            torch.save(features, output_file)
            print(f"✓ 已保存至: {output_file}")

        except Exception as e:
            print(f"✗ 处理 {dataset} 数据集时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n{'='*60}")
    print(f"✓ AAindex 特征生成完成！")
    print(f"{'='*60}\n")

    # 显示生成的文件
    print("已生成的文件:")
    aaindex_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('_aaindex.pt')])
    if aaindex_files:
        for file in aaindex_files:
            file_path = os.path.join(args.output_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB

            # 读取文件查看形状
            data = torch.load(file_path)
            print(f"  - {file} ({file_size:.2f} MB) - 形状: {data.shape}")
    else:
        print("  (未生成任何文件)")


if __name__ == '__main__':
    main()

    """
    使用示例:

    # 生成训练集的 AAindex 特征
    python generate_aaindex.py --dataset train

    # 生成验证集的 AAindex 特征
    python generate_aaindex.py --dataset valid

    # 生成测试集的 AAindex 特征
    python generate_aaindex.py --dataset test

    # 生成所有数据集的 AAindex 特征
    python generate_aaindex.py --dataset all

    # 自定义路径
    python generate_aaindex.py --dataset new_test \
        --fasta_dir /media/tgliu/hdd/qyguo/experience3/data/test_set_cd-hit_40 \
        --aaindex_file /media/tgliu/hdd/qyguo/experience3/data/aaindex/aaindex1.txt \
        --output_dir /media/tgliu/hdd/qyguo/experience3/data/aaindex
    """
