import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
from memory_module import MambaCell, TransformerCell 

def init_params(m):
    classname = m.__class__.__name__
    if "Linear" in classname:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.zero_()

# ---------- Actor‑Critic model ------------------------------------------
class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, model_type=None):
        super().__init__()

        # Flags
        self.use_memory = use_memory
        self.model_type = model_type

        # ── Image Module ──────────────────────────────────────────────────
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.ReLU(),
        )
        h, w = obs_space["image"][:2]
        self.image_embedding_size = ((h - 1) // 2 - 2) * ((w - 1) // 2 - 2) * 64

        # ── Memory Module ──────────────────────────────────────────────────
        if self.use_memory:

            if model_type == "lstm":
                self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)
            
            if model_type == "mamba":
                self.memory_rnn = MambaCell(
                    self.image_embedding_size,
                    self.semi_memory_size,
                    d_state=16,
                    d_conv=4
                )
            
            if model_type == "transformer":
                self.memory_rnn = TransformerCell(
                    self.image_embedding_size,
                    self.semi_memory_size,
                    num_heads = 4,
                    mem_len = 38,
                )

            # Torch‑ac needs to know how big the flat state is
            if isinstance(self.memory_rnn, (MambaCell, TransformerCell)):
                self._memory_size = self.memory_rnn.memory_size
            else:  # LSTMCell
                self._memory_size = 2 * self.semi_memory_size
        else:
            self._memory_size = 0

        # Joint embedding size
        self.embedding_size = self.semi_memory_size

        # Actor & Critic heads
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, action_space.n),
        )
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

        # Initialise
        self.apply(init_params)

    # API needed by torch‑ac
    @property
    def memory_size(self):
        # LSTM uses (h,c) so doubles the size, others define their own
        if not self.use_memory:
            return 0
        if self.model_type == "lstm":
            return 2 * self.semi_memory_size
        else:
            # mamba / transformer each know how big their memory is
            return self.memory_rnn.memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    # ── Forward pass ──────────────────────────────────────────────────
    def forward(self, obs, memory):
        # Image embedding
        # obs.image comes in (B, H, W, C); convert to (B, C, H, W)
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)  # flatten

        # Memory / recurrence
        if self.use_memory:
            if self.model_type == "lstm":
                # unpack (h, c) from the single memory tensor
                h_prev = memory[:, :self.semi_memory_size]
                c_prev = memory[:, self.semi_memory_size:]
                # one LSTMCell step
                h_next, c_next = self.memory_rnn(x, (h_prev, c_prev))
                embedding = h_next
                # repack (h, c) into a single tensor
                memory = torch.cat([h_next, c_next], dim=1)
            else:
                # mamba or transformer already return (embedding, new_memory)
                embedding, memory = self.memory_rnn(x, memory)
        else:
            # no recurrence
            embedding = x

        # --- Actor & Critic heads ---
        logits = self.actor(embedding)
        dist   = Categorical(logits=F.log_softmax(logits, dim=1))
        value = self.critic(embedding).squeeze(1)
        return dist, value, memory