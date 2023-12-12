

def model_fn(x, model, init_kwargs: dict = {}, call_kwargs: dict = {}):
    model = model(**init_kwargs)
    return model(x, **call_kwargs)
