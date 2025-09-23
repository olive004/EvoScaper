

import optax


def make_optimiser(learning_rate_sched: str = 'cosine_decay',
                   learning_rate = 0.1, epochs: int = 100,
                   a_decay: float = 0.0,
                   use_warmup: bool = False,
                   warmup_epochs: int = 0,
                   n_batches: int = 1,
                   method: str = 'sgd',
                   min_learning_rate=1e-6):
    opt_method = getattr(optax, method)
    if use_warmup:
        warmup_fn = optax.linear_schedule(
            init_value=min_learning_rate, end_value=learning_rate,
            transition_steps=warmup_epochs * n_batches)
            # transition_steps=warmup_epochs)
        cosine_epochs = max(epochs - warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=cosine_epochs * n_batches, 
            alpha=a_decay)
        learning_rate_scheduler = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs * n_batches])
    else:
        if learning_rate_sched == 'cosine_decay':
            learning_rate_scheduler = optax.cosine_decay_schedule(
                learning_rate, decay_steps=epochs, alpha=a_decay)
        else:
            learning_rate_scheduler = learning_rate
    optimiser = opt_method(learning_rate=learning_rate_scheduler)
    return optimiser
