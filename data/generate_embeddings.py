#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: qyguo
"""
PLM-GNN 嵌入生成脚本
生成 ESM-1b、ESM-2 (多种型号)、ProtT5、TAPE、ESM-C 和 ESM-3 嵌入用于毒力因子多分类任务

支持的模型:
    - esm1b: ESM-1b (1280-dim)
    - esm2_t6: ESM-2 6层 8M (320-dim)
    - esm2_t12: ESM-2 12层 35M (480-dim)
    - esm2_t30: ESM-2 30层 150M (640-dim)
    - esm2_t33: ESM-2 33层 650M (1280-dim)
    - esm2_t36: ESM-2 36层 3B (2560-dim)
    - esm2_t48: ESM-2 48层 15B (5120-dim)
    - prot_t5: ProtT5-XL half precision (1024-dim)
    - tape: TAPE ProteinBERT (768-dim)
    - esmc_300m: ESM-C 300M (960-dim)
    - esmc_600m: ESM-C 600M (1152-dim)
    - esm3_sm: ESM-3 Small Open (1536-dim)

使用方法:
    python generate_embeddings.py --model esm1b --dataset train
    python generate_embeddings.py --model esm2_t33 --dataset valid
    python generate_embeddings.py --model prot_t5 --dataset test
    python generate_embeddings.py --model tape --dataset train
    python generate_embeddings.py --model esmc_600m --dataset train
    python generate_embeddings.py --model esm3_sm --dataset train
    python generate_embeddings.py --model all --dataset all
"""

# ============================================================================
# 公共库导入
# ============================================================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 设置HuggingFace镜像地址（国内加速）
import sys
import argparse
import random
import torch
import gc
import re
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import esm

# 设置随机种子
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    """设置所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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


def extract_esm1b_embeddings_from_fasta(file_path, device, max_len=1022):
    """
    使用 ESM-1b 提取蛋白质序列嵌入

    ESM-1b 是一个基于Transformer的蛋白质语言模型，包含33层，650M参数
    模型在UniRef50数据集上预训练，能够生成1280维的蛋白质序列表示

    参数:
        file_path: FASTA 文件路径
        device: 计算设备 (cuda 或 cpu)
        max_len: 最大序列长度 (ESM-1b 默认最大 1024，减去特殊token为1022)

    返回:
        embeddings_with_labels: tensor of shape (num_sequences, 1281)
                               第一列为标签，后1280列为ESM-1b嵌入
    """
    print(f"\n{'='*60}")
    print(f"正在加载 ESM-1b 模型 (esm1b_t33_650M_UR50S)...")
    print(f"{'='*60}")

    # 加载预训练的ESM-1b模型和字母表
    model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 用官方 alphabet 自动构建合法字符集（单字符 token）
    valid_chars = {t for t in alphabet.all_toks if len(t) == 1}

    embeddings = []
    labels = []
    records = list(SeqIO.parse(file_path, "fasta"))

    replacement_stats = {}  # 统计字符替换情况
    warned = False  # 是否已经显示过警告

    print(f"\n正在从 {len(records)} 条序列中提取 ESM-1b 嵌入...")

    for record in tqdm(records, desc="ESM-1b 嵌入提取"):
        # 提取标签
        label = get_label_from_id(record.id)
        if label == -1:
            print(f"\n警告: 无法识别序列ID {record.id} 的标签，跳过该序列")
            continue
        labels.append(label)

        # 处理序列
        seq = str(record.seq).upper()
        if len(seq) > max_len:
            seq = seq[:max_len]  # 截断过长序列

        # 先处理 stop 符号（删除）
        if "*" in seq:
            replacement_stats["*->(removed)"] = replacement_stats.get("*->(removed)", 0) + seq.count("*")
            seq = seq.replace("*", "")

        # 再把不在 alphabet 的字符替换为 X
        cleaned = []
        for ch in seq:
            if ch not in valid_chars:
                replacement_stats[ch] = replacement_stats.get(ch, 0) + 1
                cleaned.append("X")
            else:
                cleaned.append(ch)
        seq = "".join(cleaned)

        if (not warned) and replacement_stats:
            print("\n警告: 检测到 ESM-1b 字母表外字符/stop 符号，已做清理（首次提示）")
            warned = True

        # 转换为 ESM-1b 输入格式
        tuple_seq = (record.id, seq)
        _, _, batch_tokens = batch_converter([tuple_seq])
        batch_tokens = batch_tokens.to(device)

        # 前向传播，提取第33层的表示
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=False)

        # 提取第33层的表示并取平均（去除首尾特殊token）
        token_representations = results["representations"][33]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        seq_repr = token_representations[0, 1:batch_lens[0] - 1].mean(0)
        embeddings.append(seq_repr.cpu())

        # 清理显存
        del batch_tokens, results, token_representations
        torch.cuda.empty_cache()
        gc.collect()

    # 显示序列清理统计信息
    if replacement_stats:
        print("\n序列清理统计:")
        for k, v in sorted(replacement_stats.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    # 拼接标签和嵌入向量
    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embeddings_with_labels = torch.cat([labels_tensor, embeddings_tensor], dim=1)

    print(f"\n✓ ESM-1b 嵌入提取完成，形状: {embeddings_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-1280 列: ESM-1b 嵌入 (1280维)")

    return embeddings_with_labels


def extract_esm2_embeddings_from_fasta(file_path, device, model_name="esm2_t33", max_len=7000):
    """
    使用 ESM-2 (ESM库原生API) 提取蛋白质序列嵌入

    ESM-2 是ESM-1b的改进版本，提供多种规模的模型选择
    支持更长的序列，并在更大的数据集上训练

    参数:
        file_path: FASTA 文件路径
        device: 计算设备 (cuda 或 cpu)
        model_name: ESM-2 模型名称 (esm2_t6/t12/t30/t33/t36/t48)
        max_len: 最大序列长度

    返回:
        embeddings_with_labels: tensor of shape (num_sequences, emb_dim+1)
                               第一列为标签，其余为ESM-2嵌入
    """
    # ESM-2 模型配置 (模型加载函数, 层数, 嵌入维度)
    esm2_models = {
        'esm2_t6': (esm.pretrained.esm2_t6_8M_UR50D, 6, 320),
        'esm2_t12': (esm.pretrained.esm2_t12_35M_UR50D, 12, 480),
        'esm2_t30': (esm.pretrained.esm2_t30_150M_UR50D, 30, 640),
        'esm2_t33': (esm.pretrained.esm2_t33_650M_UR50D, 33, 1280),
        'esm2_t36': (esm.pretrained.esm2_t36_3B_UR50D, 36, 2560),
        'esm2_t48': (esm.pretrained.esm2_t48_15B_UR50D, 48, 5120),
    }

    if model_name not in esm2_models:
        raise ValueError(f"未知的 ESM-2 模型: {model_name}. 可用模型: {list(esm2_models.keys())}")

    model_loader, num_layers, emb_dim = esm2_models[model_name]

    print(f"\n{'='*60}")
    print(f"正在加载 ESM-2 模型: {model_name}")
    print(f"层数: {num_layers}, 嵌入维度: {emb_dim}")
    print(f"{'='*60}")

    # 加载模型和字母表
    model, alphabet = model_loader()
    batch_converter = alphabet.get_batch_converter()
    model = model.to(device)
    model.eval()  # 设置为评估模式

    # 用官方 alphabet 自动构建合法字符集（单字符 token）
    valid_chars = {t for t in alphabet.all_toks if len(t) == 1}

    embeddings = []
    labels = []
    records = list(SeqIO.parse(file_path, "fasta"))

    replacement_stats = {}  # 统计字符替换情况
    warned = False  # 是否已经显示过警告

    print(f"\n正在从 {len(records)} 条序列中提取 {model_name} 嵌入...")

    for record in tqdm(records, desc=f"{model_name} 嵌入提取"):
        # 提取标签
        label = get_label_from_id(record.id)
        if label == -1:
            print(f"\n警告: 无法识别序列ID {record.id} 的标签，跳过该序列")
            continue
        labels.append(label)

        # 处理序列
        seq = str(record.seq).upper()
        if len(seq) > max_len:
            seq = seq[:max_len]  # 截断过长序列

        # 先处理 stop 符号（删除）
        if "*" in seq:
            replacement_stats["*->(removed)"] = replacement_stats.get("*->(removed)", 0) + seq.count("*")
            seq = seq.replace("*", "")

        # 再把不在 alphabet 的字符替换为 X
        cleaned = []
        for ch in seq:
            if ch not in valid_chars:
                replacement_stats[ch] = replacement_stats.get(ch, 0) + 1
                cleaned.append("X")
            else:
                cleaned.append(ch)
        seq = "".join(cleaned)

        if (not warned) and replacement_stats:
            print(f"\n警告: 检测到 ESM-2 字母表外字符/stop 符号，已做清理（首次提示）")
            warned = True

        # 转换为 ESM-2 输入格式
        tuple_seq = (record.id, seq)
        batch_labels, batch_strs, batch_tokens = batch_converter([tuple_seq])
        batch_tokens = batch_tokens.to(device)

        # 前向传播，提取指定层的表示
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[num_layers], return_contacts=False)

        # 提取指定层的表示并取平均（去除首尾特殊token）
        # 注意：令牌0始终是序列开始令牌，所以第一个残基的令牌是1
        token_representations = results["representations"][num_layers]
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        seq_repr = token_representations[0, 1:batch_lens[0] - 1].mean(0)
        embeddings.append(seq_repr.cpu())

        # 清理显存
        del batch_tokens, results, token_representations, batch_labels, batch_strs
        torch.cuda.empty_cache()
        gc.collect()

    # 显示序列清理统计信息
    if replacement_stats:
        print("\n序列清理统计:")
        for k, v in sorted(replacement_stats.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    # 拼接标签和嵌入向量
    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embeddings_with_labels = torch.cat([labels_tensor, embeddings_tensor], dim=1)

    print(f"\n✓ {model_name} 嵌入提取完成，形状: {embeddings_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-{emb_dim} 列: {model_name} 嵌入 ({emb_dim}维)")

    return embeddings_with_labels


def extract_prot_t5_embeddings_from_fasta(file_path, device, max_length=7000):
    """
    使用 ProtT5-XL 提取蛋白质序列嵌入

    ProtT5 是基于T5架构的蛋白质语言模型，在BFD数据库上预训练
    使用半精度浮点数以减少内存占用，生成1024维的序列表示

    参数:
        file_path: FASTA 文件路径
        device: 计算设备 (cuda 或 cpu)
        max_length: 最大序列长度

    返回:
        embeddings_with_labels: tensor of shape (num_sequences, 1025)
                               第一列为标签,后1024列为ProtT5嵌入
    """
    # ProtT5-specific imports
    from transformers import T5Tokenizer, T5EncoderModel

    print(f"\n{'='*60}")
    print(f"正在加载 ProtT5-XL 模型 (prot_t5_xl_half_uniref50-enc)...")
    print(f"{'='*60}")

    # 设置 HuggingFace 镜像（国内网络加速）
    if os.environ.get('HF_ENDPOINT') is None:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

    T5_model_path = "Rostlab/prot_t5_xl_half_uniref50-enc"

    # 加载tokenizer和模型
    tokenizer = T5Tokenizer.from_pretrained(T5_model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(T5_model_path)
    model = model.to(device)
    model.eval()  # 设置为评估模式

    embeddings = []
    labels = []
    records = list(SeqIO.parse(file_path, "fasta"))

    replacement_stats = {}  # 统计字符替换情况
    warned = False  # 是否已经显示过警告

    print(f"\n正在从 {len(records)} 条序列中提取 ProtT5 嵌入...")

    for record in tqdm(records, desc="ProtT5 嵌入提取"):
        # 提取标签
        label = get_label_from_id(record.id)
        if label == -1:
            print(f"\n警告: 无法识别序列ID {record.id} 的标签，跳过该序列")
            continue
        labels.append(label)

        # 处理序列
        seq = str(record.seq)
        if len(seq) > max_length:
            seq = seq[:max_length]  # 截断过长序列

        # 处理序列：替换稀有氨基酸为 X (U, Z, O, B 不在标准20种氨基酸中)
        original_seq = seq
        processed_seq = [" ".join(list(re.sub(r"[UZOB]", "X", seq)))]

        # 统计替换的字符
        for ch in ['U', 'Z', 'O', 'B']:
            if ch in original_seq:
                replacement_stats[ch] = replacement_stats.get(ch, 0) + original_seq.count(ch)

        if (not warned) and replacement_stats:
            print("\n警告: 检测到 ProtT5 字母表外字符，已做清理（首次提示）")
            warned = True

        # 使用tokenizer编码序列
        ids = tokenizer(processed_seq, add_special_tokens=True, padding="longest")
        input_ids = torch.tensor(ids['input_ids']).to(device)
        attention_mask = torch.tensor(ids['attention_mask']).to(device)

        # 前向传播获取嵌入
        with torch.no_grad():
            seq_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # 计算序列长度并取平均（去除特殊token）
        seq_len = attention_mask[0].sum()
        seq_output = seq_outputs.last_hidden_state[0, :seq_len - 1].mean(0)
        embeddings.append(seq_output.cpu())

        # 清理显存
        del input_ids, attention_mask, seq_outputs, seq_output
        gc.collect()
        torch.cuda.empty_cache()

    # 显示序列清理统计信息
    if replacement_stats:
        print("\n序列清理统计:")
        for k, v in sorted(replacement_stats.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    # 拼接标签和嵌入向量
    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embeddings_with_labels = torch.cat([labels_tensor, embeddings_tensor], dim=1)

    print(f"\n✓ ProtT5 嵌入提取完成，形状: {embeddings_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-1024 列: ProtT5 嵌入 (1024维)")

    return embeddings_with_labels


def extract_tape_embeddings_from_fasta(file_path, device, max_length=7000):
    """
    使用 TAPE (ProteinBERT) 提取蛋白质序列嵌入

    TAPE (Tasks Assessing Protein Embeddings) 是一个基于BERT架构的蛋白质语言模型
    在Pfam数据库上预训练，生成768维的序列表示

    参数:
        file_path: FASTA 文件路径
        device: 计算设备 (cuda 或 cpu)
        max_length: 最大序列长度

    返回:
        embeddings_with_labels: tensor of shape (num_sequences, 769)
                               第一列为标签,后768列为TAPE嵌入
    """
    # TAPE-specific imports
    try:
        from tape import ProteinBertModel, TAPETokenizer
        TAPE_AVAILABLE = True
    except ImportError:
        TAPE_AVAILABLE = False

    if not TAPE_AVAILABLE:
        raise ImportError("TAPE 未安装。请使用以下命令安装: pip install tape-proteins")

    print(f"\n{'='*60}")
    print(f"正在加载 TAPE ProteinBERT 模型 (bert-base)...")
    print(f"{'='*60}")

    # 加载TAPE模型和tokenizer
    model = ProteinBertModel.from_pretrained('bert-base')
    tokenizer = TAPETokenizer(vocab='iupac')  # iupac 是 TAPE 模型使用的词汇表
    model = model.to(device)
    model.eval()  # 设置为评估模式

    embeddings = []
    labels = []
    records = list(SeqIO.parse(file_path, "fasta"))

    replacement_stats = {}  # 统计字符替换情况
    warned = False  # 是否已经显示过警告

    print(f"\n正在从 {len(records)} 条序列中提取 TAPE 嵌入...")

    for record in tqdm(records, desc="TAPE 嵌入提取"):
        # 提取标签
        label = get_label_from_id(record.id)
        if label == -1:
            print(f"\n警告: 无法识别序列ID {record.id} 的标签，跳过该序列")
            continue
        labels.append(label)

        # 处理序列
        seq = str(record.seq)
        if len(seq) > max_length:
            seq = seq[:max_length]  # 截断过长序列

        # TAPE使用IUPAC词汇表，记录任何非标准氨基酸
        standard_aa = set('ACDEFGHIKLMNPQRSTVWY')
        for ch in seq:
            if ch not in standard_aa:
                replacement_stats[ch] = replacement_stats.get(ch, 0) + 1

        if (not warned) and replacement_stats:
            print("\n警告: 检测到 TAPE 字母表外字符（首次提示）")
            warned = True

        # 编码序列
        encoded_seq = tokenizer.encode(seq)
        token_ids = torch.tensor([np.array(encoded_seq)]).to(device)

        # 前向传播，获取模型输出
        with torch.no_grad():
            # outputs[0]表示每个残基的768D嵌入表示
            outputs = model(token_ids)[0]
            # 对所有残基的嵌入取平均，得到序列级别的表示
            seq_repr = outputs.mean(dim=1).squeeze()

        embeddings.append(seq_repr.cpu())

        # 清理显存
        del token_ids, outputs
        gc.collect()
        torch.cuda.empty_cache()

    # 显示序列清理统计信息
    if replacement_stats:
        print("\n序列清理统计:")
        for k, v in sorted(replacement_stats.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v}")

    # 拼接标签和嵌入向量
    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embeddings_with_labels = torch.cat([labels_tensor, embeddings_tensor], dim=1)

    print(f"\n✓ TAPE 嵌入提取完成，形状: {embeddings_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-768 列: TAPE 嵌入 (768维)")

    return embeddings_with_labels


def extract_esmc_embeddings_from_fasta(
    file_path,
    device,
    model_name="esmc_600m",
    max_length=7000,
):
    """
    使用 ESM-C 提取蛋白序列嵌入

    ESM-C 是ESM系列的最新版本，采用改进的架构和训练策略
    提供300M和600M两种规模的模型，分别生成960维和1152维的嵌入

    参数:
        file_path: FASTA 文件路径
        device: 计算设备 (cuda 或 cpu)
        model_name: ESM-C 模型名称 (esmc_300m 或 esmc_600m)
        max_length: 最大序列长度

    返回:
        embeddings_with_labels: Tensor (N, 1 + D)
                               第一列为标签，其余为ESM-C嵌入
    """
    # ESM-C-specific imports
    try:
        from esm.models.esmc import ESMC
        from esm.sdk.api import ESMProtein, LogitsConfig
        ESMC_AVAILABLE = True
    except ImportError:
        ESMC_AVAILABLE = False

    if isinstance(device, str):
        device = torch.device(device)

    if not ESMC_AVAILABLE:
        raise ImportError("ESM-C 未安装。请使用以下命令安装: pip install esm")

    # --------------------------------------------------
    # ESM-C 模型配置
    # --------------------------------------------------
    esmc_models = {
        "esmc_300m": 960,
        "esmc_600m": 1152,
    }

    if model_name not in esmc_models:
        raise ValueError(
            f"未知的 ESM-C 模型: {model_name}. 可用模型: {list(esmc_models.keys())}"
        )

    emb_dim = esmc_models[model_name]

    print(f"\n{'='*60}")
    print(f"正在加载 ESM-C 模型 ({model_name})...")
    print(f"嵌入维度: {emb_dim}")
    print(f"{'='*60}")

    # --------------------------------------------------
    # 加载模型（使用SDK方式）
    # --------------------------------------------------
    try:
        client = ESMC.from_pretrained(model_name).to(device)
    except Exception as e:
        raise RuntimeError(f"加载 ESM-C 模型 '{model_name}' 失败: {repr(e)}")

    # --------------------------------------------------
    # 读取 FASTA 文件
    # --------------------------------------------------
    records = list(SeqIO.parse(file_path, "fasta"))

    embeddings = []
    labels = []

    replacement_stats = {}  # 统计字符替换情况
    warned = False  # 是否已经显示过警告

    print(f"\n正在从 {len(records)} 条序列中提取 {model_name} 嵌入...")

    # --------------------------------------------------
    # 主循环：逐条处理序列
    # --------------------------------------------------
    for idx, record in enumerate(
        tqdm(records, desc=f"{model_name} 嵌入提取", ncols=90)
    ):
        label = get_label_from_id(record.id)
        if label == -1:
            print(f"\n警告: 无法识别序列ID {record.id} 的标签，跳过该序列")
            continue

        seq = str(record.seq).upper()
        if len(seq) > max_length:
            seq = seq[:max_length]  # 截断过长序列
        if len(seq) == 0:
            continue

        # 创建蛋白质对象
        protein = ESMProtein(sequence=seq)

        # 编码序列
        try:
            protein_tensor = client.encode(protein)
        except Exception as e:
            if idx < 3:  # 只显示前3个错误
                print(f"\n警告: ESM-C encode 失败 (ID={record.id}): {repr(e)}")
            continue

        # 获取嵌入表示
        try:
            out = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
        except Exception as e:
            if idx < 3:  # 只显示前3个错误
                print(f"\n警告: ESM-C logits 失败 (ID={record.id}): {repr(e)}")
            continue

        emb = getattr(out, "embeddings", None)
        if emb is None:
            continue

        # 处理嵌入维度: (1, L, D) or (L, D)
        if emb.dim() == 3:
            emb_seq = emb[0]
        elif emb.dim() == 2:
            emb_seq = emb
        else:
            continue

        # 平均池化得到序列级表示
        seq_emb = emb_seq.mean(dim=0)
        embeddings.append(seq_emb.detach().cpu())
        labels.append(float(label))

        # 清理显存
        del protein_tensor, out, emb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if len(embeddings) == 0:
        raise RuntimeError("未能提取任何 ESM-C 嵌入。")

    # 拼接标签和嵌入向量
    embeddings_tensor = torch.stack(embeddings)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embeddings_with_labels = torch.cat([labels_tensor, embeddings_tensor], dim=1)

    print(f"\n✓ {model_name} 嵌入提取完成，形状: {embeddings_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-{emb_dim} 列: {model_name} 嵌入 ({emb_dim}维)")

    return embeddings_with_labels



def extract_esm3_embeddings_from_fasta(
    file_path: str,
    device: torch.device,
    batch_size: int = 4,
    max_length: int = 7000,
    model_name: str = "esm3-sm-open-v1",
):
    """
    使用 ESM-3 提取蛋白序列嵌入

    ESM-3 是ESM系列的第三代模型，采用全新的架构设计
    支持更长的序列和更复杂的蛋白质结构预测任务
    使用序列级平均池化生成固定维度的嵌入表示

    参数:
        file_path: FASTA 文件路径
        device: 计算设备 (cuda 或 cpu)
        batch_size: 批处理大小（默认为4，可根据显存调整）
        max_length: 最大序列长度
        model_name: ESM-3 模型名称

    返回:
        embeddings_with_labels: Tensor (N, 1+D)
                               第一列为标签，其余为ESM-3嵌入
    """
    # ESM-3-specific imports
    try:
        from esm.models.esm3 import ESM3
        from esm.sdk.api import ESMProtein, ESM3InferenceClient, LogitsConfig
        ESM3_AVAILABLE = True
    except ImportError as e:
        ESM3_AVAILABLE = False
        raise ImportError(f"导入 ESM-3 SDK 模块失败: {repr(e)}")

    if isinstance(device, str):
        device = torch.device(device)

    print(f"\n{'='*60}")
    print(f"正在加载 ESM-3 模型 ({model_name})...")
    print(f"{'='*60}")

    # --------------------------------------------------
    # 加载 ESM-3 模型
    # --------------------------------------------------
    try:
        client: ESM3InferenceClient = ESM3.from_pretrained(model_name).to(device)
    except Exception as e:
        raise RuntimeError(f"加载 ESM-3 模型 '{model_name}' 失败: {repr(e)}")

    # --------------------------------------------------
    # 读取 FASTA 文件
    # --------------------------------------------------
    records = list(SeqIO.parse(file_path, "fasta"))

    embeddings = []
    labels = []

    replacement_stats = {}  # 统计字符替换情况
    warned = False  # 是否已经显示过警告

    print(f"\n正在从 {len(records)} 条序列中提取 ESM-3 嵌入...")

    # --------------------------------------------------
    # 主循环：逐条处理序列
    # --------------------------------------------------
    for idx, rec in enumerate(
        tqdm(records, desc="ESM-3 嵌入提取", ncols=90)
    ):
        label = get_label_from_id(rec.id)
        if label == -1:
            # 给出明确警告
            print(f"\n警告: 无法识别序列ID {rec.id} 的标签，跳过该序列")
            continue

        seq = str(rec.seq).upper()
        if len(seq) > max_length:
            seq = seq[:max_length]  # 截断过长序列
        if len(seq) == 0:
            continue

        # 创建蛋白质对象
        protein = ESMProtein(sequence=seq)

        # 编码序列
        try:
            protein_tensor = client.encode(protein)
        except Exception as e:
            if idx < 3:  # 只显示前3个错误
                print(f"\n警告: ESM-3 encode 失败 (ID={rec.id}): {repr(e)}")
            continue

        # 获取嵌入表示
        try:
            out = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True)
            )
        except Exception as e:
            if idx < 3:  # 只显示前3个错误
                print(f"\n警告: ESM-3 logits 失败 (ID={rec.id}): {repr(e)}")
            continue

        emb = getattr(out, "embeddings", None)
        if emb is None:
            continue

        # 处理嵌入维度: (1, L, D) or (L, D)
        if emb.dim() == 3:
            emb_seq = emb[0]
        elif emb.dim() == 2:
            emb_seq = emb
        else:
            continue

        # 平均池化得到序列级表示
        seq_emb = emb_seq.mean(dim=0)
        embeddings.append(seq_emb.detach().cpu())
        labels.append(float(label))

        # 清理显存
        del protein_tensor, out, emb
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    # --------------------------------------------------
    # 收尾检查与打印
    # --------------------------------------------------
    if len(embeddings) == 0:
        raise RuntimeError("未能提取任何 ESM-3 嵌入。请检查输入 FASTA 文件和标签。")

    # 拼接标签和嵌入向量
    embeddings_tensor = torch.stack(embeddings)   # (N, D)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    embeddings_with_labels = torch.cat([labels_tensor, embeddings_tensor], dim=1)

    emb_dim = embeddings_tensor.shape[1]

    print(f"\n✓ ESM-3 嵌入提取完成，形状: {embeddings_with_labels.shape}")
    print(f"  - 第 0 列: 标签")
    print(f"  - 第 1-{emb_dim} 列: ESM-3 嵌入 ({emb_dim}维)")

    return embeddings_with_labels


def main():
    """
    主函数：处理命令行参数并生成蛋白质嵌入

    支持多种蛋白质语言模型和多个数据集的批量处理
    """
    parser = argparse.ArgumentParser(description='为 PLM-GNN 生成蛋白质嵌入')
    parser.add_argument('--model', type=str,
                        choices=['esm1b', 'esm2_t6', 'esm2_t12', 'esm2_t30',
                                'esm2_t33', 'esm2_t36', 'esm2_t48', 'prot_t5',
                                'tape', 'esmc_300m', 'esmc_600m', 'esm3_sm', 'all'],
                        default='esm1b',
                        help='选择用于嵌入提取的模型')
    parser.add_argument('--dataset', type=str, choices=['train', 'valid', 'test', 'new_test', 'all'],
                        default='all',
                        help='选择要处理的数据集')
    parser.add_argument('--fasta_dir', type=str, default='./fasta',
                        help='包含 FASTA 文件的目录')
    parser.add_argument('--output_dir', type=str, default='./embedding',
                        help='保存嵌入的目录')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用的设备 (cuda 或 cpu)')
    parser.add_argument('--max_len_esm', type=int, default=1022,
                        help='ESM-1b 的最大序列长度')
    parser.add_argument('--max_len_esm2', type=int, default=7000,
                        help='ESM-2 的最大序列长度')
    parser.add_argument('--max_len_t5', type=int, default=7000,
                        help='ProtT5 的最大序列长度')
    parser.add_argument('--max_len_tape', type=int, default=7000,
                        help='TAPE 的最大序列长度')
    parser.add_argument('--max_len_esmc', type=int, default=7000,
                        help='ESM-C 的最大序列长度')
    parser.add_argument('--max_len_esm3', type=int, default=2000,
                        help='ESM-3 的最大序列长度')
    parser.add_argument('--esm3_model_path', type=str, default=None,
                        help='ESM-3 模型权重路径（可选，未提供则使用默认路径）')
    parser.add_argument('--esm3_batch_size', type=int, default=1,
                        help='ESM-3 的批处理大小（默认为1以避免显存溢出）')

    # 解析命令行参数
    args, unknown = parser.parse_known_args()

    # 设置随机种子以保证可复现性
    set_seed()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 确定要处理的数据集
    if args.dataset == 'all':
        datasets = ['train', 'valid', 'test']
    else:
        datasets = [args.dataset]

    # 确定要使用的模型
    if args.model == 'all':
        models = ['esm1b', 'esm2_t6', 'esm2_t12', 'esm2_t30',
                 'esm2_t33', 'esm2_t36', 'esm2_t48', 'prot_t5',
                 'tape', 'esmc_300m', 'esmc_600m', 'esm3_sm']
    else:
        models = [args.model]

    # 文件名映射
    file_mapping = {
        'train': 'train_set.fasta',
        'valid': 'valid_set.fasta',
        'test': 'test_set.fasta',
        'new_test': 'new_test_set.fasta'
    }

    print(f"\n{'='*60}")
    print(f"PLM-GNN 蛋白质嵌入生成")
    print(f"{'='*60}")
    print(f"模型: {', '.join(models)}")
    print(f"数据集: {', '.join(datasets)}")
    print(f"设备: {args.device}")
    print(f"FASTA 目录: {args.fasta_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"{'='*60}\n")

    # 处理每个数据集和模型组合
    for dataset in datasets:
        fasta_file = os.path.join(args.fasta_dir, file_mapping[dataset])

        if not os.path.exists(fasta_file):
            print(f"警告: 文件 {fasta_file} 不存在，跳过")
            continue

        print(f"\n{'#'*60}")
        print(f"正在处理 {dataset} 数据集: {fasta_file}")
        print(f"{'#'*60}")

        for model_idx, model_name in enumerate(models):
            print(f"\n[{model_idx+1}/{len(models)}] 正在生成 {model_name} 嵌入...")

            try:
                if model_name == 'esm1b':
                    embeddings = extract_esm1b_embeddings_from_fasta(
                        fasta_file, args.device, max_len=args.max_len_esm
                    )
                    output_file = os.path.join(args.output_dir, f'{dataset}_esm1b.pt')

                elif model_name.startswith('esm2_'):
                    embeddings = extract_esm2_embeddings_from_fasta(
                        fasta_file, args.device, model_name=model_name,
                        max_len=args.max_len_esm2
                    )
                    output_file = os.path.join(args.output_dir, f'{dataset}_{model_name}.pt')

                elif model_name == 'prot_t5':
                    embeddings = extract_prot_t5_embeddings_from_fasta(
                        fasta_file, args.device, max_length=args.max_len_t5
                    )
                    output_file = os.path.join(args.output_dir, f'{dataset}_prot_t5.pt')

                elif model_name == 'tape':
                    embeddings = extract_tape_embeddings_from_fasta(
                        fasta_file, args.device, max_length=args.max_len_tape
                    )
                    output_file = os.path.join(args.output_dir, f'{dataset}_tape.pt')

                elif model_name in ['esmc_300m', 'esmc_600m']:
                    embeddings = extract_esmc_embeddings_from_fasta(
                        fasta_file, args.device, model_name=model_name,
                        max_length=args.max_len_esmc
                    )
                    output_file = os.path.join(args.output_dir, f'{dataset}_{model_name}.pt')

                elif model_name == 'esm3_sm':
                    embeddings = extract_esm3_embeddings_from_fasta(
                        fasta_file, args.device,
                        batch_size=args.esm3_batch_size,
                        max_length=args.max_len_esm3
                    )
                    output_file = os.path.join(args.output_dir, f'{dataset}_esm3_sm.pt')

                else:
                    print(f"未知模型: {model_name}，跳过")
                    continue

                # 保存嵌入
                torch.save(embeddings, output_file)
                print(f"✓ 已保存至: {output_file}")

                # 清理内存
                del embeddings
                gc.collect()
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"✗ 处理 {model_name} 在 {dataset} 数据集时出错: {str(e)}")
                import traceback
                traceback.print_exc()
                continue

    print(f"\n{'='*60}")
    print(f"✓ 嵌入生成完成！")
    print(f"{'='*60}\n")

    # 显示生成的文件
    print("已生成的文件:")
    embedding_files = sorted([f for f in os.listdir(args.output_dir) if f.endswith('.pt')])
    if embedding_files:
        for file in embedding_files:
            file_path = os.path.join(args.output_dir, file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {file} ({file_size:.2f} MB)")
    else:
        print("  (未生成任何文件)")


if __name__ == '__main__':
    main()

    """
    # 生成训练集的 ESM-1b 嵌入
    python generate_embeddings.py --model esm1b --dataset train

    # 生成验证集的 ESM-2-t33 嵌入
    python generate_embeddings.py --model esm2_t33 --dataset valid

    # 生成低冗余测试集的 ProtT5 嵌入
    python generate_embeddings.py --model prot_t5 --dataset new_test --fasta_dir ./test_set_cd-hit_40

    # 生成训练集的 TAPE 嵌入
    python generate_embeddings.py --model tape --dataset train



    ### 注意！！！！
    # ESM-3、ESM-C 目前需要 hf 的 token，
    # 1. 登录 https://huggingface.co/EvolutionaryScale/esm3-sm-open-v1 获取 token
    #       export HF_TOKEN=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
    # 2. 终端执行：hf auth login



    # 生成训练集的 esmc_600m 嵌入
    python generate_embeddings.py --model esmc_600m --dataset train

    # 生成训练集的 ESM-3 嵌入
    python generate_embeddings.py --model esm3_sm --dataset train
    """