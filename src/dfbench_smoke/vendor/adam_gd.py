# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/gradient_based/adam_gd.py
import optax
from jaxtyping import Array, Float

from dfbench import OptimizationAlgorithm, Objective


class AdamGD(OptimizationAlgorithm):
    """Standard Adam optimizer with global-norm gradient clipping."""

    algorithm_str = "adam_gd"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        learning_rate: float = 0.1,
        **adam_kwargs,
    ) -> None:
        obj = objective
        # Adam expects an unbounded search space; the Objective sigmoid-maps
        # back to the physical bounds before each evaluation.
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        # Start from a random unbounded point unless one was supplied.
        params = init_params if init_params is not None else obj.random_params_unbounded()

        # Clip gradients to a global L2 norm of 1.0 to stabilise early updates,
        # then apply a standard Adam step.
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        state = optimizer.init(params)

        # Compile the value+grad path before the clock starts (free).
        obj.warmup_value_and_grad()
        obj.start_logging()

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            # Optional early stopping when no improvement for `patience` evals.
            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)