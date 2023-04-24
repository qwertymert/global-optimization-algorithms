# Author       : Mert Gülşen

import numpy as np
import time


def naive_random_search(f, x_init, objective="min", neighborhood="normal", iters=100):
    """
    Naive random search algorithm for global optimization problems.
    :param f: Objective function to optimize
    :param x_init: Initial x vector
    :param objective: maximize ("max") or minimize ("min")
    :param neighborhood: probability distribution to choose neighbor point
    :param iters: number of iterations before termination
    :return: optimum x, optimum cost, execution time
    """

    time1 = time.time()  # start time
    cost, x = f(x_init), x_init  # initialize cost and starting point

    for i in range(iters):  # iterate for given number of iterations
        if neighborhood == "normal":  # create a random point from given neighbourhood distribution
            x_new = [np.random.normal(x[j], 1) for j in range(len(x))]
        else:
            x_new = [np.random.uniform(x[j]-0.5, x[j]+0.5) for j in range(len(x))]
        new_cost = f(x_new)
        if objective == "min":  # compare new cost with current one according to objective
            condition = new_cost < cost
        else:
            condition = new_cost > cost

        if condition:  # update current point and cost
            x = x_new
            cost = new_cost

    return x, cost, time.time()-time1


def simulated_annealing(f, x_init, objective="min", gamma=0.1, neighborhood="normal", iters=100):
    """
    Simulated annealing algorithm for global optimization problems.
    :param f: Objective function to optimize
    :param x_init: Initial x vector
    :param objective: maximize ("max") or minimize ("min")
    :param gamma: problem-dependent constant to control cooling schedule
    :param neighborhood: probability distribution to choose neighbor point
    :param iters: number of iterations before termination
    :return: optimum x, optimum cost, execution time
    """

    time1 = time.time()  # start time
    cost, x, t = f(x_init), x_init, gamma/np.log(2)  # initialize cost, starting point and gamma

    for i in range(iters):
        if neighborhood == "normal":  # create a random point from given neighbourhood distribution
            x_new = [np.random.normal(x[j], 1) for j in range(len(x))]
        else:
            x_new = [np.random.uniform(x[j] - 0.5, x[j] + 0.5) for j in range(len(x))]

        new_cost = f(x_new)
        if objective == "min":  # apply probability formula according to objective (max or min)
            condition = np.random.random() < min([1, np.exp(-(f(x_new) - f(x))/t)])
        else:
            condition = np.random.random() < min([1, np.exp((f(x_new) - f(x))/t)])

        if condition:  # update current point and cost
            x = x_new
            cost = new_cost

        t = gamma/np.log(i+1 + 2)  # update cooling schedule

    return x, cost, time.time()-time1


def particle_swarm(f, n=2, objective="min", d=10, w=0.5, c1=2, c2=2, iters=100):
    """
    Particle swarm optimization algorithm for global optimization problems.
    :param f: Objective function to optimize
    :param n: Dimension of input vector to objective function
    :param objective: maximize ("max") or minimize ("min")
    :param d: number of particles in the swarm
    :param w: inertial constant to control velocity of particles
    :param c1: cognitive component to control influence of personal best
    :param c2: social component to control influence of global best
    :param iters: number of iterations before termination
    :return:  global best point (optimum x), global best cost, execution time
    """

    condition = objective == "min"
    time1 = time.time()

    swarm = np.random.uniform(-10, 10, (d, n))  # randomly initialize swarm of d particles
    velocities = np.random.uniform(-1, 1, (d, n))  # randomly initialize velocities of particles
    p_best = swarm  # initialize personal best points of particles

    if condition:  # update global best particle point
        g_best = swarm[np.argmin(f([swarm[:, i] for i in range(n)]))].reshape(1, -1)
    else:
        g_best = swarm[np.argmax(f([swarm[:, i] for i in range(n)]))].reshape(1, -1)

    for i in range(iters):
        # randomly generate r and s vectors
        r, s = np.random.uniform(0, 1, (d, n)), np.random.uniform(0, 1, (d, n))
        # update velocities by applying update formula
        velocities = w*velocities + (c1*r * (p_best - swarm)) + c2*s * (g_best - swarm)
        swarm = swarm + velocities  # update swarm positions

        if condition:  # get idx of better positioned particles compared to previous ones
            idx = f([swarm[:, i] for i in range(n)]) < f([p_best[:, i] for i in range(n)])
        else:
            idx = f([swarm[:, i] for i in range(n)]) > f([p_best[:, i] for i in range(n)])
        p_best[idx] = swarm[idx]  # update personal best positions of particles

        if condition:  # get best position of current swarm and update global best point
            swarm_best = swarm[np.argmin(f([swarm[:, i] for i in range(n)]))].reshape(1, -1)
            g_best = swarm_best if f(swarm_best[0]) < f(g_best[0]) else g_best
        else:
            swarm_best = swarm[np.argmax(f([swarm[:, i] for i in range(n)]))].reshape(1, -1)
            g_best = swarm_best if f(swarm_best[0]) > f(g_best[0]) else g_best

    return g_best[0], f(g_best[0]), time.time()-time1


def genetic_algorithm(f, dim, n_members=40, omega=3, bits=16, pm=0.05, iters=100):
    """
    Genetic algorithm for global maximization problems.
    :param f: Objective function to optimize
    :param dim: Dimension of input vector to objective function
    :param n_members: number of members in the population
    :param omega: Range constraint for creating population
    :param bits: Number of bits to encode member values
    :param pm: Probability of mutation in genes
    :param iters: Number of iterations before termination
    :return: the best chromosome (optimum x), maximized best cost, execution time
    """

    time1 = time.time()  # start time

    # initialize a population randomly in the constraint range
    population = np.random.uniform(-omega, omega, (n_members, dim))

    # encoding operation
    domain = np.linspace(-omega, omega, 100)  # create domain values for interpolation
    target = np.linspace(0, 2**bits-1, 100)  # create target values
    interp = np.interp(population, domain, target).astype(int)  # interpolate population values for binary conversion
    # encode population values as binary strings
    encoded = np.array(["".join([np.binary_repr(num, bits) for num in member]) for member in interp])

    # initialize global best chromosome from initial population
    best_chromosome = population[np.argmax(f([population[:, i] for i in range(dim)]))]

    for i in range(iters):  # iterations
        evaluate = f([population[:, i] for i in range(dim)])  # get cost values for each member
        best_of_population = population[np.argmax(evaluate)]  # get best of current population
        # update global best chromosome if needed
        best_chromosome = best_of_population if f(best_chromosome) < f(best_of_population) else best_chromosome

        # shift cost values of population to make them positive in order to calculate probabilities
        evaluate = evaluate + abs(min(evaluate)) + 1 if min(evaluate) < 0 else evaluate
        probs = evaluate / np.sum(evaluate)  # calculate probability of each chromosome
        mating_pool = np.random.choice(encoded, n_members, True, probs)  # create a mating pool from probabilities

        # choose parent chromosomes for cross over operations
        pairs_idx = np.random.choice(np.arange(n_members), np.random.randint(1, n_members//2)*2).reshape(-1, 2)
        pairs = mating_pool[pairs_idx]

        for k in range(len(pairs)):  # cross-over operation
            cs = np.random.randint(1, bits)  # crossing site
            parent1, parent2 = list(pairs[k, 0]), list(pairs[k, 1])  # convert binary string to list
            parent1[:cs], parent2[:cs] = parent2[:cs], parent1[:cs]  # cross-over
            mating_pool[pairs_idx[k, 0]] = "".join(parent1)  # replace mating pool parents with their offspring
            mating_pool[pairs_idx[k, 1]] = "".join(parent2)

        for k in range(len(mating_pool)):  # mutation operation
            # get genes to be mutated with mutation probability
            mutation_idx = np.where(np.random.uniform(0, 1, bits*2) < pm)[0]
            chromosome = np.array(list(mating_pool[k]))
            for b in mutation_idx:  # mutate genes
                if chromosome[b] == "0":
                    chromosome[b] = "1"
                else:
                    chromosome[b] = "0"
            mating_pool[k] = "".join(chromosome)  # update chromosome

        encoded = mating_pool  # update encoded population
        # interpolate encoded population to normal values and get next generation of population
        population = np.array([np.interp([int(m[:bits], 2), int(m[bits:], 2)], target, domain) for m in mating_pool])

    return best_chromosome, f(best_chromosome), time.time()-time1
