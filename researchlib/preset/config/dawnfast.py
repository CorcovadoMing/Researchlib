def Dawnfast(**kwargs):
    default_kwargs = dict(
        epochs=10, 
        lr=1, 
        warmup=2, 
        flatten=2, 
        flatten_lr=0.1, 
        fp16=True, 
        ema_freq=5, 
        bias_scale=64,
        batch_size=512,
        monitor=['acc'],
        init='default'
    )
    default_kwargs.update(kwargs)
    return default_kwargs