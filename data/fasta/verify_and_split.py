#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集验证和分离脚本
验证valid_test.fasta包含valid_set，并分离出test_set
"""

from Bio import SeqIO
import os

# 文件路径
fasta_dir = '/media/tgliu/hdd/qyguo/experience3/PLM-GNN-main/data/fasta'
train_file = os.path.join(fasta_dir, 'train_set.fasta')
valid_file = os.path.join(fasta_dir, 'valid_set.fasta')
valid_test_file = os.path.join(fasta_dir, 'valid_test.fasta')
test_file = os.path.join(fasta_dir, 'test_set.fasta')

print("="*60)
print("数据集验证和分离")
print("="*60)

# 读取所有数据集
print("\n[1/5] 读取数据集...")
train_records = list(SeqIO.parse(train_file, 'fasta'))
valid_records = list(SeqIO.parse(valid_file, 'fasta'))
valid_test_records = list(SeqIO.parse(valid_test_file, 'fasta'))

print(f"  - train_set: {len(train_records)} 条序列")
print(f"  - valid_set: {len(valid_records)} 条序列")
print(f"  - valid_test: {len(valid_test_records)} 条序列")

# 提取ID集合
train_ids = set(record.id for record in train_records)
valid_ids = set(record.id for record in valid_records)
valid_test_ids = set(record.id for record in valid_test_records)

print(f"\n[2/5] 验证1: 检查valid_set是否是valid_test的子集...")
valid_in_valid_test = valid_ids.issubset(valid_test_ids)
if valid_in_valid_test:
    print(f"  ✓ 验证通过: valid_set的所有{len(valid_ids)}条序列都在valid_test中")
else:
    missing = valid_ids - valid_test_ids
    print(f"  ✗ 验证失败: 有{len(missing)}条序列不在valid_test中")
    print(f"  缺失的序列ID: {list(missing)[:5]}...")

# 分离测试集
print(f"\n[3/5] 分离测试集...")
test_ids = valid_test_ids - valid_ids
print(f"  - valid_test总数: {len(valid_test_ids)}")
print(f"  - valid_set数量: {len(valid_ids)}")
print(f"  - test_set数量: {len(test_ids)}")

# 创建test_set.fasta
test_records = [record for record in valid_test_records if record.id in test_ids]
print(f"\n  提取到 {len(test_records)} 条测试集序列")

# 保存test_set.fasta
SeqIO.write(test_records, test_file, 'fasta')
print(f"  ✓ 已保存到: {test_file}")

# 验证类别分布
print(f"\n[4/5] 验证类别分布...")

def get_class_distribution(records):
    """统计类别分布"""
    from collections import Counter
    classes = [record.id[:7] for record in records]
    return Counter(classes)

train_dist = get_class_distribution(train_records)
valid_dist = get_class_distribution(valid_records)
test_dist = get_class_distribution(test_records)

print(f"\n  训练集类别分布:")
for cls, count in sorted(train_dist.items()):
    print(f"    {cls}: {count}")

print(f"\n  验证集类别分布:")
for cls, count in sorted(valid_dist.items()):
    print(f"    {cls}: {count}")

print(f"\n  测试集类别分布:")
for cls, count in sorted(test_dist.items()):
    print(f"    {cls}: {count}")

# 验证数据集之间没有重叠
print(f"\n[5/5] 验证2: 检查train、valid、test之间是否有重叠...")

# 重新读取test_set
test_records_new = list(SeqIO.parse(test_file, 'fasta'))
test_ids_new = set(record.id for record in test_records_new)

# 检查重叠
train_valid_overlap = train_ids & valid_ids
train_test_overlap = train_ids & test_ids_new
valid_test_overlap = valid_ids & test_ids_new

print(f"\n  检查结果:")
print(f"    train ∩ valid: {len(train_valid_overlap)} 条重叠")
print(f"    train ∩ test:  {len(train_test_overlap)} 条重叠")
print(f"    valid ∩ test:  {len(valid_test_overlap)} 条重叠")

if train_valid_overlap:
    print(f"    ✗ train和valid有重叠: {list(train_valid_overlap)[:5]}...")
else:
    print(f"    ✓ train和valid无重叠")

if train_test_overlap:
    print(f"    ✗ train和test有重叠: {list(train_test_overlap)[:5]}...")
else:
    print(f"    ✓ train和test无重叠")

if valid_test_overlap:
    print(f"    ✗ valid和test有重叠: {list(valid_test_overlap)[:5]}...")
else:
    print(f"    ✓ valid和test无重叠")

# 总结
print(f"\n{'='*60}")
print("总结")
print(f"{'='*60}")
print(f"✓ 数据集统计:")
print(f"  - train_set.fasta: {len(train_ids)} 条序列")
print(f"  - valid_set.fasta: {len(valid_ids)} 条序列")
print(f"  - test_set.fasta:  {len(test_ids_new)} 条序列")
print(f"  - 总计: {len(train_ids) + len(valid_ids) + len(test_ids_new)} 条序列")

all_no_overlap = (len(train_valid_overlap) == 0 and
                  len(train_test_overlap) == 0 and
                  len(valid_test_overlap) == 0)

if all_no_overlap:
    print(f"\n✓ 验证通过: 三个数据集之间没有重叠")
else:
    print(f"\n✗ 验证失败: 数据集之间存在重叠")

print(f"\n✓ test_set.fasta 已成功创建!")
print(f"{'='*60}\n")
