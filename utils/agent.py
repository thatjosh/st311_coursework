import torch

import utils
from .other import device
from model import ACModel


class Agent:
    """
    Wrapper used during training.
    Works with both recurrent and non‑recurrent checkpoints.
    """
    def __init__(self, obs_space, action_space, model_dir,
                 argmax=False, num_envs=1,
                 use_memory=False, use_text=False,
                 model_type=None):

        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)

        # ── load model ───────────────────────────────────────────
        self.acmodel = ACModel(obs_space, action_space,
                               use_memory=use_memory,
                               use_text=use_text,
                               model_type=model_type)
        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(device).eval()

        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

        # ── bookkeeping ──────────────────────────────────────────
        self.argmax = argmax
        self.num_envs = num_envs
        self.has_mem = self.acmodel.recurrent

        if self.has_mem:
            self.memories = torch.zeros(self.num_envs,
                                        self.acmodel.memory_size,
                                        device=device)
        else:
            self.memories = None

    # ----------------------------------------------------------
    def get_actions(self, obss):
        """
        obss: list of raw observations (len = num_envs)
        returns: numpy array of actions, shape (num_envs,)
        """
        batch = self.preprocess_obss(obss, device=device)

        mem_in = self.memories if self.has_mem else None
        with torch.no_grad():
            dist, _, mem_out = self.acmodel(batch, mem_in)

        # choose actions
        if self.argmax:
            actions = dist.probs.argmax(1, keepdim=True)
        else:
            actions = dist.sample()

        # update recurrent state
        if self.has_mem:
            self.memories = mem_out

        return actions.cpu().numpy()

    def get_action(self, obs):
        """Single‑env convenience wrapper."""
        return self.get_actions([obs])[0]

    # ----------------------------------------------------------
    def analyze_feedbacks(self, rewards, dones):
        """
        Called once per *vectorised* step to mask memories on episode end.
        rewards, dones: lists/arrays with len == num_envs
        """
        if self.has_mem:
            masks = 1 - torch.as_tensor(dones, dtype=torch.float,
                                        device=device).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        """Single‑env convenience wrapper."""
        self.analyze_feedbacks([reward], [done])