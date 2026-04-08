from abc import ABC, abstractmethod
from model import GPT2Like, GPTConfig
from gpt_config_builder import GPTConfigBuilder


class AbstractModelFactory(ABC):
  @abstractmethod
  def create_model(self, model_type: str) -> GPT2Like:
    """Return a fully-constructed GPT2Like model."""
    ...


class ModelFactory(AbstractModelFactory):
  """
  Factory for GPT2Like models.

  Supported types
  ---
  'standard' : baseline model (256-dim, 4 layers) used for training.
  'small'    : tiny model (128-dim, 2 layers) for quick tests.
  """

  # The Builder runs at class definition time, so invalid configs are caught immediately on import rather than silently at runtime.
  _CONFIGS: dict[str, GPTConfig] = {
    "standard": (GPTConfigBuilder()
                      .set_vocab_size(8000)
                      .set_block_size(128)
                      .set_model_dim(256)
                      .set_layers(4)
                      .set_heads(4)
                      .set_dropout(0.1)
                      .build()),
    "small":    (GPTConfigBuilder()
                      .set_vocab_size(8000)
                      .set_block_size(64)
                      .set_model_dim(128)
                      .set_layers(2)
                      .set_heads(2)
                      .set_dropout(0.1)
                      .build()),
}

  def create_model(self, model_type: str) -> GPT2Like:
    if model_type not in self._CONFIGS:
      raise ValueError(
        f"Unknown model type '{model_type}'. "
        f"Choose from: {list(self._CONFIGS.keys())}"
      )
    return GPT2Like(self._CONFIGS[model_type])