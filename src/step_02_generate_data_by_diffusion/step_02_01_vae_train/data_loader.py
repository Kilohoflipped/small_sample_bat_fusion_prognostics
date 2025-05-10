"""电池老化循环数据加载模块，包含Dataset类定义、数据预处理及标准化功能"""

import pandas as pd
import numpy as np
import torch
import torch.utils.data as torch_data

class BatteryDataset(torch_data.Dataset):
    """电池老化循环数据集类，用于处理时间序列数据的加载与标准化

        Attributes:
            battery_ids: 电池ID列表，形状为[N,]
            sequences: 标准化后的电池循环序列数据列表，形状为[N, seq_len]
            conditions: 标准化后的环境条件参数列表，形状为[N, condition_dim]
            lengths: 每个电池循环序列的实际长度列表，形状为[N,]
    """

    def __init__(self, battery_ids, sequences, conditions, lengths):
        self.battery_ids = battery_ids
        self.sequences = sequences
        self.conditions = conditions
        self.lengths = lengths

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].clone().detach().to(dtype=torch.float32).unsqueeze(-1)
        cond = self.conditions[idx].clone().detach().to(dtype=torch.float32)
        length = self.lengths[idx]
        battery_id = self.battery_ids[idx]
        return seq, cond, length, battery_id

class BatteryDataLoader:
    """Battery data loader, encapsulating preprocessing, normalization, and DataLoader creation
    电池数据加载器，封装数据预处理、标准化及DataLoader创建逻辑

    Attributes:
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        dataset: Constructed Dataset instance
    """

    def __init__(self, csv_path, cond_bounds, exclude_battery_ids, shuffle, batch_size=4):
        """
        Args:
            csv_path (str): Path to the preprocessed CSV file
            batch_size (int): Batch size, default is 4
            shuffle (bool): Whether to shuffle data
        """
        self.csv_path = csv_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.processor = BatteryDataProcessor(csv_path, cond_bounds, exclude_battery_ids)
        self.dataset = BatteryDataset(
            sequences=self.processor.processed_data['sequences'],
            conditions=self.processor.processed_data['conditions'],
            lengths=self.processor.processed_data['lengths'],
            battery_ids=self.processor.processed_data['battery_ids']
        )

        self.target_mean = self.processor.processed_data['target_mean']
        self.target_std = self.processor.processed_data['target_std']
        self.raw_sequences = self.processor.processed_data['raw_sequences']

    def create_loader(self):
        """
        创建配置好的DataLoader
        """
        if not self.dataset:
            raise ValueError("Dataset not initialized")
        return torch_data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=collate_fn,
            num_workers=0
        )

class BatteryDataProcessor:
    def __init__(self, csv_path, cond_bounds, exclude_battery_ids=None):
        self.csv_path = csv_path
        self.exclude_battery_ids = exclude_battery_ids or []

        self.cond_bounds = cond_bounds

        self.raw_data = self._load_data()
        self.processed_data = self._process_data()

    def _load_data(self):
        df = pd.read_csv(self.csv_path)
        if self.exclude_battery_ids:
            df = df[~df['battery_id'].isin(self.exclude_battery_ids)]

        grouped = df.groupby('battery_id')

        raw_data = {
            'sequences': [group['target'].values for _, group in grouped],
            'conditions': [
                [
                    self._parse_rate(group['charge_rate'].iloc[0]),
                    self._parse_rate(group['discharge_rate'].iloc[0]),
                    float(group['temperature'].iloc[0]),
                    float(group['pressure'].iloc[0]),
                    float(group['dod'].iloc[0])
                ] for _, group in grouped
            ],
            'lengths': [len(group) for _, group in grouped],
            'battery_ids': [group['battery_id'].iloc[0] for _, group in grouped]
        }

        return raw_data

    def _process_data(self):
        sequences = self.raw_data['sequences']
        conditions = self.raw_data['conditions']
        sequences, target_mean, target_std = self._normalize_sequences(sequences)
        conditions = self._normalize_conditions(conditions)

        processed_data = {
            'sequences': sequences,
            'conditions': conditions,
            'lengths': self.raw_data['lengths'],
            'battery_ids': self.raw_data['battery_ids'],
            'target_mean': target_mean,
            'target_std': target_std,
            'raw_sequences': self.raw_data['sequences']
        }

        return processed_data

    @staticmethod
    def _normalize_sequences(sequences):
        seq_tensor = torch.tensor(np.concatenate(sequences), dtype=torch.float32)
        mean, std = seq_tensor.mean(), seq_tensor.std()
        return [(torch.tensor(seq, dtype=torch.float32) - mean) / std for seq in sequences], mean, std

    def _normalize_conditions(self,conditions):
        cond_tensor = torch.tensor(conditions, dtype=torch.float32)
        cond_bound_tensor = torch.tensor(self.cond_bounds, dtype=torch.float32)

        min_vals = cond_bound_tensor[:, 0]
        max_vals = cond_bound_tensor[:, 1]
        range_vals = max_vals - min_vals

        norm_conditions = (cond_tensor - min_vals) / range_vals
        return [c for c in norm_conditions]

    @staticmethod
    def _parse_rate(rate):
        if isinstance(rate, str):
            rate = rate.replace('C', '').strip()
            try:
                return float(rate)
            except ValueError:
                raise ValueError(f"Invalid rate format: '{rate}'")
        else:
            raise ValueError(f"Invalid rate format: '{rate}'")

def collate_fn(batch):
    """
    整理批次数据，填充序列至最大长度，堆叠条件和长度张量

    Args:
        batch (List[Tuple]): Batch data from Dataset, each element is a (seq, cond, length) tuple
                             where seq is (seq_len, 1), cond is (condition_dim,), length is int

    Returns:
        padded_seqs (Tensor): Padded sequence tensor, shape (batch_size, max_len, 1)
        conditions (Tensor): Stacked condition tensor, shape (batch_size, condition_dim)
        lengths (Tensor): Tensor of actual sequence lengths, shape (batch_size,)
    """
    seqs, conditions, lengths, battery_ids = zip(*batch)

    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

    battery_ids = [battery_ids[i] for i in sorted_indices]
    seqs = [seqs[i] for i in sorted_indices]
    conditions = [conditions[i] for i in sorted_indices]
    lengths = [lengths[i] for i in sorted_indices]

    padded_seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    conditions = torch.stack(conditions)
    lengths = torch.tensor(lengths, dtype=torch.long, device=padded_seqs.device)

    return padded_seqs, conditions, lengths, battery_ids
