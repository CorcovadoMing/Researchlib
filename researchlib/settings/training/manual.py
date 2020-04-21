def Manual(**kwargs):
    default_kwargs = dict(
        epochs=1, 
        lr=1, 
        policy='overwrite',
        warmup=0, 
        batch_size=128,
        accum_grad=1,
        fp16=True,
    )
    default_kwargs.update(kwargs)
    return default_kwargs