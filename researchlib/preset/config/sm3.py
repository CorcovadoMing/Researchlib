def SM3(**kwargs):
    default_kwargs = dict(
        epochs=300, 
        lr=1e-1, 
        policy='fixed',
        warmup=5, 
        batch_size=128,
        accum_grad=1,
        fp16=True,
        final_anneal=30,
        init='xavier',
    )
    default_kwargs.update(kwargs)
    return default_kwargs