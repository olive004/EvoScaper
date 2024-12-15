

import optax


def make_optimiser(learning_rate_sched: str,
                   learning_rate, epochs: int = None,
                   l2_reg_alpha: float = None,
                   use_warmup: bool = False,
                   warmup_epochs: int = 0,
                   n_batches: int = 1,
                   method: str = 'sgd'):
    opt_method = getattr(optax, method)
    if use_warmup:
        warmup_fn = optax.linear_schedule(
            init_value=0., end_value=learning_rate,
            transition_steps=warmup_epochs * n_batches)
        cosine_epochs = max(epochs - warmup_epochs, 1)
        cosine_fn = optax.cosine_decay_schedule(
            init_value=learning_rate,
            decay_steps=cosine_epochs * n_batches)
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_epochs * n_batches])
        optimiser = opt_method(learning_rate=schedule_fn)
    else:
        if learning_rate_sched == 'cosine_decay':
            learning_rate_scheduler = optax.cosine_decay_schedule(
                learning_rate, decay_steps=epochs, alpha=l2_reg_alpha)
        else:
            learning_rate_scheduler = learning_rate
        optimiser = opt_method(learning_rate=learning_rate_scheduler)
    return optimiser
