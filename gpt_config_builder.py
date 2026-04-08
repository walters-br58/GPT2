from model import GPTConfig


class GPTConfigBuilder:
  """
  Fluent builder for GPTConfig.
  """

  def __init__(self):
      # Defaults
      self._vocab_size: int   = 8000
      self._block_size: int   = 128
      self._d_model:    int   = 256
      self._n_layers:   int   = 4
      self._n_heads:    int   = 4
      self._dropout:    float = 0.1
      self._bias:       bool  = False

  # Setter methods (each returns self for chaining)

  def set_vocab_size(self, v: int) -> "GPTConfigBuilder":
    if v < 1:
      raise ValueError("vocab_size must be >= 1")
    self._vocab_size = v
    return self

  def set_block_size(self, v: int) -> "GPTConfigBuilder":
    if v < 1:
      raise ValueError("block_size must be >= 1")
    self._block_size = v
    return self

  def set_model_dim(self, v: int) -> "GPTConfigBuilder":
    if v < 1:
      raise ValueError("d_model must be >= 1")
    self._d_model = v
    return self

  def set_layers(self, v: int) -> "GPTConfigBuilder":
    if v < 1:
      raise ValueError("n_layers must be >= 1")
    self._n_layers = v
    return self

  def set_heads(self, v: int) -> "GPTConfigBuilder":
    if v < 1:
      raise ValueError("n_heads must be >= 1")
    self._n_heads = v
    return self

  def set_dropout(self, v: float) -> "GPTConfigBuilder":
    if not (0.0 <= v < 1.0):
      raise ValueError("dropout must be in [0, 1)")
    self._dropout = v
    return self

  def set_bias(self, v: bool) -> "GPTConfigBuilder":
    self._bias = v
    return self

  # Build
  def build(self) -> GPTConfig:
    """Validate cross-field constraints, then construct and return the config."""
    if self._d_model % self._n_heads != 0:
      raise ValueError(
        f"d_model ({self._d_model}) must be divisible by "
        f"n_heads ({self._n_heads})"
      )
    return GPTConfig(
      vocab_size = self._vocab_size,
      block_size = self._block_size,
      d_model    = self._d_model,
      n_layers   = self._n_layers,
      n_heads    = self._n_heads,
      dropout    = self._dropout,
      bias       = self._bias,
    )