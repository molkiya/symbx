import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 1) Discrete rules ("formula chunks")
# -----------------------------

@dataclass
class Rule:
    name: str
    fn: Callable[[torch.Tensor], torch.Tensor]

def add(k: float) -> Rule:
    return Rule(f"+{k}", lambda x: x + k)

def mul(k: float) -> Rule:
    return Rule(f"*{k}", lambda x: x * k)

# Basic rule set (can be extended)
RULES: List[Rule] = [
    add(1.0), add(2.0), add(3.0),
    mul(2.0), mul(3.0)
]

STOP_ACTION = "STOP"  # special action for program termination

# -----------------------------
# 2) Helper functions
# -----------------------------

def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return a / (b + eps)

def make_state(cur: torch.Tensor, target: torch.Tensor, t: int, device) -> torch.Tensor:
    """
    Extended state representation:
      current value, target value, delta, ratio, |delta|, sign(delta), step index
    Helps the policy distinguish modes: "multiplicative step" vs "additive fine-tuning".
    """
    delta = target - cur
    ratio = safe_div(target, cur).clamp(-10.0, 10.0)
    sign_delta = torch.sign(delta)
    feats = torch.tensor([
        cur.item(),
        target.item(),
        delta.item(),
        ratio.item(),
        abs(delta).item(),
        sign_delta.item(),
        float(t)
    ], device=device)
    return feats

# -----------------------------
# 3) Environment
# -----------------------------

class ProgramEnv:
    def __init__(self, rules: List[Rule], max_steps: int = 5, step_penalty: float = 0.01):
        self.rules = rules
        self.max_steps = max_steps
        self.step_penalty = step_penalty

    def rollout(self, policy, A: float, X: float, train: bool = True):
        """
        Executes one rollout:
          - starts from input A
          - applies up to max_steps rules chosen by the policy
          - returns prediction, actions, logprobs, program names, reward, mse
        """
        device = next(policy.parameters()).device
        cur = torch.tensor([A], dtype=torch.float32, device=device)
        target = torch.tensor([X], dtype=torch.float32, device=device)

        actions, logprobs, names, entropies = [], [], [], []
        prev_err = torch.mean((cur - target) ** 2)
        total_reward = 0.0

        for t in range(self.max_steps):
            state = make_state(cur, target, t, device)
            logits = policy(state)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)

            if train:
                a = dist.sample()
            else:
                a = torch.argmax(probs)

            lp = dist.log_prob(a)
            ent = - (probs * (probs.clamp_min(1e-9)).log()).sum()

            a = int(a.item())
            actions.append(a)
            logprobs.append(lp)
            entropies.append(ent)

            if a == len(self.rules):
                names.append(STOP_ACTION)
                break

            rule = self.rules[a]
            names.append(rule.name)
            cur = rule.fn(cur)

            # step shaping reward: improvement in error
            new_err = torch.mean((cur - target) ** 2)
            step_reward = (prev_err - new_err).detach() - self.step_penalty
            total_reward += step_reward
            prev_err = new_err

        mse = torch.mean((cur - target) ** 2)
        exact_bonus = 0.1 if torch.allclose(cur, target, atol=1e-6) else 0.0  # бонус за точное совпадение
        reward = (total_reward - mse + exact_bonus).detach()

        return cur.detach(), actions, logprobs, names, reward, mse.item(), entropies

# -----------------------------
# 4) Policy network
# -----------------------------

class Policy(nn.Module):
    def __init__(self, num_actions: int, hidden: int = 64):
        super().__init__()
        # input: [cur, target, delta, ratio, |delta|, sign(delta), step]
        self.net = nn.Sequential(
            nn.Linear(7, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)

# -----------------------------
# 5) Synthetic dataset generator
# -----------------------------

def apply_program(A: float, prog: List[Rule]) -> float:
    x = A
    for r in prog:
        x = float(r.fn(torch.tensor([x])).item())
    return x

def sample_hidden_program(rules: List[Rule], max_len: int) -> List[Rule]:
    L = random.randint(1, max_len)
    return [random.choice(rules) for _ in range(L)]

def make_dataset(rules: List[Rule], max_len: int, num_tasks: int = 200, A_range=(-5, 5)) -> List[Tuple[float, float, List[str]]]:
    tasks = []
    for _ in range(num_tasks):
        prog = sample_hidden_program(rules, max_len=max_len)
        A = random.uniform(*A_range)
        X = apply_program(A, prog)
        tasks.append((A, X, [r.name for r in prog]))
    return tasks

# -----------------------------
# 6) Pretraining with imitation learning
# -----------------------------

def pretrain_imitation(policy, optimizer, tasks, rules, epochs=3):
    """
    Pretraining by supervised imitation:
      - use ground truth programs
      - minimize cross-entropy for correct actions at each step
    """
    policy.train()
    for _ in range(epochs):
        random.shuffle(tasks)
        for (A, X, true_prog_names) in tasks:
            device = next(policy.parameters()).device
            cur = torch.tensor([A], dtype=torch.float32, device=device)
            target = torch.tensor([X], dtype=torch.float32, device=device)
            ce_loss = 0.0
            for t, name in enumerate(true_prog_names):
                state = make_state(cur, target, t, device)
                logits = policy(state)
                y = torch.tensor([next(i for i, r in enumerate(rules) if r.name == name)], device=device)
                ce_loss = ce_loss + nn.CrossEntropyLoss()(logits.unsqueeze(0), y)
                cur = rules[y.item()].fn(cur)
            optimizer.zero_grad()
            ce_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

# -----------------------------
# 7) Training loop
# -----------------------------

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(42)
    torch.manual_seed(42)

    env = ProgramEnv(RULES, max_steps=5, step_penalty=0.01)
    num_actions = len(RULES) + 1  # + STOP
    policy = Policy(num_actions=num_actions, hidden=64).to(device)
    opt = optim.Adam(policy.parameters(), lr=3e-3)

    train_tasks = make_dataset(RULES, max_len=3, num_tasks=400)
    val_tasks = make_dataset(RULES, max_len=3, num_tasks=80)

    # imitation warm-up
    pretrain_imitation(policy, opt, train_tasks[:200], RULES, epochs=3)

    baseline = 0.0
    gamma_bl = 0.95
    entropy_coef = 0.01

    epochs = 40
    batch_size = 32

    for ep in range(1, epochs + 1):
        random.shuffle(train_tasks)
        ep_losses, ep_rewards, ep_mse = [], [], []
        for i in range(0, len(train_tasks), batch_size):
            batch = train_tasks[i:i + batch_size]
            batch_loss = 0.0
            batch_rewards = []
            for (A, X, _) in batch:
                pred, acts, lps, names, reward, mse, ents = env.rollout(policy, A, X, train=True)
                advantage = reward - baseline
                logprob_sum = torch.stack(lps).sum()
                entropy_mean = torch.stack(ents).mean()
                loss = -advantage * logprob_sum - entropy_coef * entropy_mean
                batch_loss += loss
                batch_rewards.append(reward.item())
                ep_mse.append(mse)

            batch_loss = batch_loss / max(1, len(batch))
            opt.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            mean_r = sum(batch_rewards) / max(1, len(batch_rewards))
            baseline = gamma_bl * baseline + (1 - gamma_bl) * mean_r

            ep_losses.append(batch_loss.item())
            ep_rewards.append(mean_r)

        if ep % 5 == 0 or ep == 1:
            with torch.no_grad():
                val_mse = []
                for (A, X, _) in val_tasks:
                    pred, acts, lps, names, reward, mse, ents = env.rollout(policy, A, X, train=False)
                    val_mse.append(mse)
            print(f"[epoch {ep:02d}] loss={sum(ep_losses)/len(ep_losses):.4f} "
                  f"train_mse={sum(ep_mse)/len(ep_mse):.4f} "
                  f"val_mse={sum(val_mse)/len(val_mse):.4f} "
                  f"baseline={baseline:.4f}")

    # Demo tasks
    print("\n=== DEMO ===")
    demo_tasks = [
        ([mul(3.0), add(2.0), mul(2.0)], 1.5),
        ([add(3.0), mul(2.0)], -2.0),
        ([mul(2.0), mul(3.0)], 4.0),
    ]
    for true_prog, A in demo_tasks:
        X = apply_program(A, true_prog)
        pred, acts, lps, names, reward, mse, ents = env.rollout(policy, A, X, train=False)
        print(f"\nA={A:.3f}  target X={X:.3f}")
        print(f"agent program: {'; '.join(names)}")
        print(f"pred={pred.item():.6f}  mse={mse:.6e}  reward={reward.item():.4f}")
        print(f"true program:  {'; '.join([r.name for r in true_prog])}")
    print("\nDone.")

# -----------------------------
# 8) Main entry
# -----------------------------

if __name__ == "__main__":
    train()
