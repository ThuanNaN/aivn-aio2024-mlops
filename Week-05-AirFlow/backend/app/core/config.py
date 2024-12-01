from dataclasses import dataclass, field

@dataclass
class BTC_Config:
    features: list = field(default_factory=lambda: ['Open', 'High', 'Low', 'Volume'])
    target: str = 'Price'
    features_scaler_path: str = "../DATA/artifacts/btc_model/v0.1/features_scaler.pkl"
    target_scaler_path: str = "../DATA/artifacts/btc_model/v0.1/target_scaler.pkl"
    model_path: str = "../DATA/artifacts/btc_model/v0.1/model.pth"
    input_size: int = 5
    hidden_size: int = 128
    output_size: int = 1
    num_layers: int = 2

