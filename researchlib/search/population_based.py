import torch

def po_meta_heuristic_(x, objective, epoch, mode, accept, opt, opt_tune):
    records = []
    x_fitness = torch.zeros(x.size(0)).cuda()
    child_fitness = torch.zeros(x.size(0)).cuda()
    child = torch.zeros_like(x).cuda()
    uniform = torch.distributions.uniform.Uniform(-0.1, 0.1)
    x_record = None
    fitness_record = None
    
    # 1. fitness
    for (i, data) in enumerate(x):
        x_fitness[i] = objective(data)
    best_fitness, index = torch.max(x_fitness, 0)
    if fitness_record is None:
        fitness_record = best_fitness
    if max(best_fitness, fitness_record) == best_fitness: 
        x_record = x[index]
        fitness_record = best_fitness
    
    for _ in range(epoch):
        # 1. fitness
        for (i, data) in enumerate(x):
            x_fitness[i] = objective(data)
        _, index = torch.max(x_fitness, 0)
        x_record = x[index]
        
        # 2. shuffle
        r = torch.randperm(x.size(0))
        x = x[r]

        # 3. Cross-over
        for i in range(0, x.size(0), 2):
            # 2-cut crossover
            c = torch.randint(0, x.size(1), [1])
            child[i] = torch.cat((x[i, :c], x[i+1, c:]))
            child[i+1] = torch.cat((x[i, c:], x[i+1, :c]))

        for (i, data) in enumerate(child):
            child_fitness[i] = objective(data)

        x_fitness, x_index = x_fitness.sort(descending=True)
        child_fitness, child_index = child_fitness.sort(descending=True)
        x = x[x_index]
        child = child[child_index]
        x[int(x.size(0)/2):] = child[:int(x.size(0)/2)] 
        
        # 4. Mutation
        x += uniform.sample(x.size()).cuda()
        
        # 5. fitness
        for (i, data) in enumerate(x):
            x_fitness[i] = objective(data)
        best_fitness, index = torch.max(x_fitness, 0)
        if max(best_fitness, fitness_record) == best_fitness: 
            x_record = x[index]
            fitness_record = best_fitness
            
        # 6. Eliteness
        x[0] = x_record
        records.append(fitness_record)
    
    return x_record, fitness_record, records
    

# Interface
# -----------------------------------------------------------

def _GeneticAlgorithm(x, objective, epoch, population=20):
    x = x.repeat(population).view(population, -1)
    return po_meta_heuristic_(x, objective, epoch, None, None, None, None)