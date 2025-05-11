import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    """使用孤立森林清洗数据的类。"""

    def __init__(self, features, contamination=0.05, random_state=42):
        """
        初始化 DataCleaner。

        Args:
            features (list): 用于异常检测的特征列表。
            contamination (float): 预期异常比例，默认为 0.05。
            random_state (int): 随机种子，确保可重复性，默认为 42。
        """
        self.features = features
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()

    def clean_with_isolation_forest(self, df):
        """
        按 battery_id 使用孤立森林清洗数据。

        Args:
            df (pd.DataFrame): 输入数据，包含电池老化数据。

        Returns:
            tuple: (df_cleaned, df_with_anomaly)
                - df_cleaned: 清洗后的数据。
                - df_with_anomaly: 包含异常标记的原始数据（anomaly 列：1 正常，-1 异常）。
        """
        df_with_anomaly = df.copy()
        df_cleaned = pd.DataFrame()

        for bid in df['battery_id'].unique():
            df_bid = df[df['battery_id'] == bid].copy()
            if len(df_bid) < 2:
                continue
            X = df_bid[self.features]
            X_scaled = self.scaler.fit_transform(X)
            iso = IsolationForest(contamination=self.contamination, random_state=self.random_state)
            df_bid['anomaly'] = iso.fit_predict(X_scaled)
            df_with_anomaly.loc[df_bid.index, 'anomaly'] = df_bid['anomaly']
            df_clean_normal = df_bid[df_bid['anomaly'] == 1].copy()
            df_cleaned = pd.concat([df_cleaned, df_clean_normal], axis=0)

        df_cleaned = df_cleaned.drop(columns=['anomaly'])
        return df_cleaned, df_with_anomaly
