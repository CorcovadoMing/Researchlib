def Cosine(**kwargs):
    default_kwargs = dict(
        epochs=150, 
        lr=1e-1, 
        policy='cosine',
        warmup=5,
        batch_size=128,
        accum_grad=1,
        fp16=True,
        init='xavier',
    )
    default_kwargs.update(kwargs)
    return default_kwargs
