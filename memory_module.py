import torch
import torch.nn as nn
from mamba import Mamba, MambaConfig

class LSTMCellWrapper(nn.Module):
    """
    A thin adapter around nn.LSTMCell, it
        - accepts (x, memory) and
        - returns (embedding, new_memory)
    where memory and new_memory are flat tensors shaped (B, 2*hidden_size)
    that pack (h, c) together.
    """

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.cell = nn.LSTMCell(input_size, hidden_size)
        self.hidden_size = hidden_size
        self.memory_size = 2 * hidden_size

    def forward(self, x_t: torch.Tensor,
                flat_state: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x_t        : (B, input_size)
        flat_state : (B, memory_size)  OR  None / empty for zero-state

        Returns
        y_t        : (B, hidden_size) – the new hidden state h_t
        flat_new   : (B, memory_size) – concatenated [h_t ⧺ c_t]
        """
        B = x_t.size(0)

        # --- unpack previous state (or build zeros) --------------------
        if flat_state is None or flat_state.numel() == 0:
            h_prev = x_t.new_zeros(B, self.hidden_size)
            c_prev = x_t.new_zeros(B, self.hidden_size)
        else:
            h_prev = flat_state[:, :self.hidden_size]
            c_prev = flat_state[:, self.hidden_size:]

        # one LSTM step
        h_next, c_next = self.cell(x_t, (h_prev, c_prev))

        # flatten & detach to keep a clean computation graph next step
        flat_new = torch.cat([h_next, c_next], dim=1).detach()
        return h_next, flat_new


class TransformerCell(nn.Module):
    """
    Single‑layer Transformer block with an LSTMCell‑like interface:
        (x_t, flat_state) → (y_t, flat_state_new)

    `flat_state` flattens the last `mem_len` hidden states so that torch‑ac
    can mask/zero them with shape (B, memory_size).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_heads: int = 4,
        mem_len: int = 36,
        ff_mult: int = 2,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mem_len = mem_len

        # Token‑wise input projection 
        self.in_proj = nn.Linear(input_size, hidden_size, bias=False)

        # Single multi‑head attention core
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=False,          # we’ll use (S, B, H) layout
        )

        # Position‑wise feed‑forward (MLP)
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, ff_mult * hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(ff_mult * hidden_size, hidden_size, bias=False),
        )

        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

        # Update memory size
        self.memory_size = mem_len * hidden_size   # torch‑ac needs this

    def forward(
        self,
        x_t: torch.Tensor,               # (B, input_size)
        flat_state: torch.Tensor | None  # (B, memory_size) or None/empty
    ):
        """
        Returns:
            y_t       (B, hidden_size)
            new_state (B, memory_size)  – detached, ready for next step
        """
        B = x_t.size(0)
        x = self.in_proj(x_t)            # (B, H)

        # Unflatten the cached hidden‑state tape
        if flat_state is None or flat_state.numel() == 0:
            # start with an empty sequence (length 0)
            mem = torch.empty(B, 0, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            mem = flat_state.view(B, self.mem_len, self.hidden_size)

        # Causal attention for the current token only
        # build K/V sequence: [past, current]
        kv = torch.cat([mem, x.unsqueeze(1)], dim=1)    # (B, L+1, H)
        kv = kv.transpose(0, 1)                         # (L+1, B, H)

        q  = x.unsqueeze(0)                             # (1,  B, H)
        attn_out, _ = self.attn(q, kv, kv, need_weights=False)
        h = attn_out.squeeze(0)                         # (B, H)

        # residual + MLP
        h = self.ln1(x + h)         # first residual branch
        y_t = self.ln2(h + self.ff(h))  # second residual branch

        # roll the memory tape & flatten for next step
        new_mem = torch.cat([mem, y_t.unsqueeze(1)], dim=1)
        if new_mem.size(1) > self.mem_len:              # keep last mem_len
            new_mem = new_mem[:, -self.mem_len:, :]
        flat_new = new_mem.reshape(B, -1).detach()      # (B, memory_size)
        return y_t, flat_new


class MambaCell(nn.Module):
    """
    Single‑layer Mamba block with an LSTMCell‑like interface:
        (x_t, flat_state) → (y_t, flat_state_new)
    `flat_state` is a 2‑D tensor so torch‑ac can mask it with (B, memory_size).
    """
    def __init__(self, input_size: int, hidden_size: int, d_state: int = 16, d_conv: int = 4):
        super().__init__()
        self.hidden_size = hidden_size

        # Input projection to Mamba d_model
        self.in_proj = nn.Linear(input_size, hidden_size, bias=False)

        # Single‑layer Mamba
        cfg = MambaConfig(
            d_model=hidden_size,
            n_layers=1,
            d_state=d_state,
            d_conv=d_conv,
        )
        self.core = Mamba(cfg)

        # Pre‑compute shapes for flattening the SSM cache
        self.d_inner = cfg.d_inner           # 2 * d_model
        self._flat_h = self.d_inner * d_state
        self._flat_inp = self.d_inner * (d_conv - 1)
        self.memory_size = self._flat_h + self._flat_inp

    def forward(self, x_t: torch.Tensor, flat_state: torch.Tensor | None):
        """
        x_t        : (B, input_size)
        flat_state : (B, memory_size) or None / empty
        returns
        y_t        : (B, hidden_size)
        new_state  : (B, memory_size)
        """
        B = x_t.size(0)
        x_t = self.in_proj(x_t)         # (B, d_model)

        # Unflatten cached state
        if flat_state is None or flat_state.numel() == 0:
            h = torch.zeros(B, self.d_inner, self.core.config.d_state, device=x_t.device, dtype=x_t.dtype)
            inp = torch.zeros(B, self.d_inner, self.core.config.d_conv - 1, device=x_t.device, dtype=x_t.dtype)
        else:
            h = flat_state[:, :self._flat_h].view(B, self.d_inner, self.core.config.d_state)
            inp = flat_state[:, self._flat_h:].view(B, self.d_inner, self.core.config.d_conv - 1)

        # Step through the single Mamba layer
        y_t, (h_new, inp_new) = self.core.layers[0].step(x_t, (h, inp))

        # Re‑flatten & detach
        flat_new = torch.cat(
            (h_new.reshape(B, -1), inp_new.reshape(B, -1)), dim=1
        ).detach()
        return y_t, flat_new