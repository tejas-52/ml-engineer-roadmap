
import numpy as np
import wandb

# Synthetic dataset: y = 3x + 2 + noise
rng = np.random.default_rng(42)
X = rng.uniform(-2, 2, size=(500, 1))
y = 3 * X + 2 + rng.normal(0, 0.2, size=(500, 1))

config = {
    "learning_rate": 0.05,
    "epochs": 200,
    "seed": 42,
    "dataset_size": len(X),
}

with wandb.init(
    project="ml-engineer-roadmap",
    config=config,
) as run:
    weights = np.zeros((1, 1))
    bias = 0.0

    for epoch in range(config["epochs"]):
        predictions = X @ weights + bias
        error = predictions - y
        loss = float(np.mean(error**2))

        dw = (2 / len(X)) * (X.T @ error)
        db = float(2 * np.mean(error))

        weights -= config["learning_rate"] * dw
        bias -= config["learning_rate"] * db

        run.log({
            "epoch": epoch,
            "mse": loss,
            "weight": float(weights[0, 0]),
            "bias": bias,
        })

    print(f"Learned weight: {weights[0, 0]:.3f}")
    print(f"Learned bias: {bias:.3f}")
    print(f"Final MSE: {loss:.5f}")
