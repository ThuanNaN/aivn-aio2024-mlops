from dataclasses import dataclass, field

@dataclass
class BTC_Config:
    features: list = field(default_factory=lambda: ['Open', 'High', 'Low', 'Volume'])
    target: str = 'Price'
    version: str = 'v0.1'
    features_scaler_path: str = field(init=False)
    target_scaler_path: str = field(init=False)
    model_path: str = field(init=False)
    input_size: int = 5
    hidden_size: int = 128
    output_size: int = 1
    num_layers: int = 2

    def __post_init__(self):
        self.features_scaler_path = f"../DATA/artifacts/btc_model/{self.version}/features_scaler.pkl"
        self.target_scaler_path = f"../DATA/artifacts/btc_model/{self.version}/target_scaler.pkl"
        self.model_path = f"../DATA/artifacts/btc_model/{self.version}/model.pth"


@dataclass
class Gold_Config:
    features: list = field(default_factory=lambda: ['Open', 'High', 'Low', 'Volume'])
    target: str = 'Price'
    