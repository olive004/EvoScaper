




def init_model(model, rng, x, cond):
    model_fn = partial(VAE_fn, enc_layers=enc_layers, dec_layers=dec_layers, decoder_head=x.shape[-1], 
                    #  HIDDEN_SIZE=HIDDEN_SIZE, USE_SIGMOID_DECODER=USE_SIGMOID_DECODER, enc_init='RandomNormal', dec_init='RandomNormal')
                    HIDDEN_SIZE=HIDDEN_SIZE, decoder_activation_final=jax.nn.sigmoid if USE_SIGMOID_DECODER else jax.nn.leaky_relu, 
                    enc_init=ENC_INIT, dec_init=DEC_INIT, activation=get_activation_fn(ACTIVATION))
    # model = hk.transform(model_fn)
    model_t = hk.multi_transform(model_fn)
    dummy_x = jax.random.normal(PRNG, x.shape)
    dummy_cond = jax.random.normal(PRNG, cond.shape)
    params = model_t.init(PRNG, dummy_x, dummy_cond, deterministic=False)

    return model.init(rng, x, cond)