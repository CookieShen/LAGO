import numpy as np
import pandas as pd
import argparse
import logging
from tqdm import tqdm
from typing import Tuple, Dict

# 配置专业的日志输出
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class FinTechEnv:
    """Semi-synthetic offline RTB environment injected with adversarial 'Risk Cliff'."""

    def __init__(self, grid_size: int = 10, num_samples: int = 100000, seed: int = 42):
        self.grid_size = grid_size
        self.num_samples = num_samples
        np.random.seed(seed)
        logging.info(f"Initializing FinTech Environment: {num_samples} samples, {grid_size}x{grid_size} grid.")
        self.df = self._generate_data()

    def _generate_data(self) -> pd.DataFrame:
        # 1. Build Traffic Density Matrix (W)
        W = np.ones((self.grid_size, self.grid_size))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if i >= 7 and j <= 2:
                    W[i, j] *= 8.0  # High intent, low qual (Toxic)
                elif i >= 7 and j >= 7:
                    W[i, j] *= 0.8  # Premium but rare
                elif i <= 3 and j >= 5:
                    W[i, j] *= 1.5  # Low intent, safe
        P_grid = W / W.sum()

        # 2. Build Approval Rate Matrix (R)
        R = np.zeros((self.grid_size, self.grid_size))
        x = np.arange(self.grid_size) / (self.grid_size - 1)
        base_approval = 0.05 + 0.83 * (x ** 3)
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                R[i, j] = np.clip(base_approval[j] * np.exp(-0.1 * i), 0.005, 0.99)

        # 3. Generate User-Level Data
        flat_indices = np.random.choice(self.grid_size ** 2, size=self.num_samples, p=P_grid.flatten())
        df = pd.DataFrame({
            'user_id': np.arange(self.num_samples),
            'grid_i': flat_indices // self.grid_size,
            'grid_j': flat_indices % self.grid_size
        })

        df['p_int'] = (df['grid_i'] + np.random.uniform(0, 1, self.num_samples)) / self.grid_size
        df['p_qual'] = (df['grid_j'] + np.random.uniform(0, 1, self.num_samples)) / self.grid_size
        df['true_approval_prob'] = R[df['grid_i'], df['grid_j']]

        # Credit Line generation (Non-linear asset quality)
        df['credit_line'] = 1000 + 29000 * (df['p_qual'] ** 2.5)
        df['credit_line'] *= np.random.uniform(0.8, 1.2, self.num_samples)

        # Market Price
        df['market_price'] = df['p_int'] * 10.0 + np.random.uniform(2.0, 4.0, self.num_samples)

        return df

    def step(self, bid_prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Executes the auction and simulates the delayed credit funnel."""
        market_price = self.df['market_price'].values
        win_rate = 1 / (1 + np.exp(-(bid_prices - market_price)))

        is_won = np.random.binomial(1, win_rate)
        cost = is_won * market_price * np.random.uniform(0.9, 1.0, self.num_samples)
        is_applied = is_won * np.random.binomial(1, self.df['p_int'])
        is_approved = is_applied * np.random.binomial(1, self.df['true_approval_prob'])
        granted_credit = is_approved * self.df['credit_line']

        return cost, is_applied, is_approved, granted_credit


class BaseAgent:
    """Base class for all auto-bidding algorithms."""

    def __init__(self, name: str, base_cpc: float = 15.0):
        self.name = name
        self.base_cpc = base_cpc

    def generate_bids(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def update(self, cost: np.ndarray, appr: np.ndarray, target_cpa: float, df: pd.DataFrame = None):
        pass


class FlatBidAgent(BaseAgent):
    def generate_bids(self, df: pd.DataFrame) -> np.ndarray:
        return df['p_int'].values * self.base_cpc


class UnconstrainedDRLAgent(BaseAgent):
    def generate_bids(self, df: pd.DataFrame) -> np.ndarray:
        return (df['p_int'].values ** 1.1) * self.base_cpc * 1.3


class PIDAgent(BaseAgent):
    def __init__(self, base_cpc: float = 15.0, lr: float = 0.001):
        super().__init__("PID-Constrained", base_cpc)
        self.alpha = 1.0
        self.lr = lr

    def generate_bids(self, df: pd.DataFrame) -> np.ndarray:
        return df['p_int'].values * self.base_cpc * self.alpha

    def update(self, cost: np.ndarray, appr: np.ndarray, target_cpa: float, df: pd.DataFrame = None):
        cpa = cost.sum() / (appr.sum() + 1e-5)
        self.alpha = np.clip(self.alpha - self.lr * (cpa - target_cpa), 0.4, 1.5)


class LAGOSolver(BaseAgent):
    """The Operations Research (OR) engine for LAGO."""

    def __init__(self, grid_size: int = 10, base_cpc: float = 15.0, lr_b: float = 0.002, lr_lambda: float = 0.0005):
        super().__init__("LAGO (Ours)", base_cpc)
        self.matrix = np.ones((grid_size, grid_size))
        self.shadow_price = 0.0
        self.lr_b = lr_b
        self.lr_lambda = lr_lambda

    def generate_bids(self, df: pd.DataFrame) -> np.ndarray:
        mults = self.matrix[df['grid_i'], df['grid_j']]
        return df['p_int'].values * self.base_cpc * mults

    def update(self, cost: np.ndarray, appr: np.ndarray, target_cpa: float, df: pd.DataFrame):
        cpa = cost.sum() / (appr.sum() + 1e-5)

        # Calculate empirical gradients grouped by state space
        df_temp = pd.DataFrame({'grid_i': df['grid_i'], 'grid_j': df['grid_j'], 'cost': cost, 'appr': appr})
        stats = df_temp.groupby(['grid_i', 'grid_j']).sum().reset_index()

        for _, row in stats.iterrows():
            i, j = int(row['grid_i']), int(row['grid_j'])
            grad = (1 + self.shadow_price * target_cpa) * row['appr'] - self.shadow_price * row['cost']
            self.matrix[i, j] = np.clip(self.matrix[i, j] + self.lr_b * grad, 0.1, 3.0)

        # Dual ascent for shadow price
        self.shadow_price = max(0.0, self.shadow_price + self.lr_lambda * (cpa - target_cpa))


def main(args):
    env = FinTechEnv(num_samples=args.samples)

    agents = [
        FlatBidAgent("Flat-Bid (Baseline)"),
        UnconstrainedDRLAgent("Unconstrained DRL"),
        PIDAgent(),
        LAGOSolver()
    ]

    results = {agent.name: {'CPA': 0, 'Volume': 0, 'Approve_Rate': 0, 'SCR': 0} for agent in agents}

    logging.info(f"Starting simulation for {args.epochs} epochs with Target CPA = {args.target_cpa}")

    for epoch in tqdm(range(args.epochs), desc="Optimizing Agents"):
        for agent in agents:
            bids = agent.generate_bids(env.df)
            cost, app, appr, cred = env.step(bids)

            # Agent self-update (if applicable)
            agent.update(cost, appr, args.target_cpa, env.df)

            # Record final epoch metrics
            if epoch == args.epochs - 1:
                results[agent.name]['CPA'] = cost.sum() / (appr.sum() + 1e-5)
                results[agent.name]['Volume'] = appr.sum()
                results[agent.name]['Approve_Rate'] = (appr.sum() / (app.sum() + 1e-5)) * 100
                results[agent.name]['SCR'] = cost.sum() / (cred.sum() + 1e-5)

    # Output formatting
    logging.info("Simulation Complete. Formatting results...")
    df_results = pd.DataFrame(results).T.reset_index()
    df_results.columns = ['Strategy', 'Actual CPA', 'Vol (V_credit)', 'R_approve', 'SCR']
    df_results['R_approve'] = df_results['R_approve'].apply(lambda x: f"{x:.2f}%")
    df_results['SCR'] = df_results['SCR'].apply(lambda x: f"{x * 100:.2f}%")

    print("\n" + "=" * 60)
    print(f"=== Final Steady-State Performance (Target CPA = {args.target_cpa}) ===")
    print("=" * 60)
    print(df_results.round(2).to_string(index=False))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LAGO Offline Simulation Environment")
    parser.add_argument('--samples', type=int, default=100000, help='Number of simulated users')
    parser.add_argument('--target_cpa', type=float, default=200.0, help='Global Target CPA constraint')
    parser.add_argument('--epochs', type=int, default=15, help='Number of optimization iterations')
    args = parser.parse_args()

    main(args)