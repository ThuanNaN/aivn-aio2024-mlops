from dataclasses import dataclass, field

@dataclass
class BTC_Config:
    features: list = field(default_factory=lambda: ['Open', 'High', 'Low', 'Volume'])
    target: str = 'Price'
    features_scaler_path: str = "features_scaler.pkl"
    target_scaler_path: str = "target_scaler.pkl"
    registered_name: str = "BTC_RNN"
    model_alias: str = "production"
