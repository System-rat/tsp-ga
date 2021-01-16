using Random
import Base

struct Node
    x::Int
    y::Int
end

struct Individual
    genome::Vector{Node}
    fitness::Float64
    distance::Float64
end

Base.max(g1::Individual, g2::Individual)::Individual = g1.fitness > g2.fitness ? g1 : g2

Base.isless(g1::Individual, g2::Individual)::Bool = g1.fitness < g2.fitness

function Base.show(io, individual::Individual)
end

distancebetween(n1::Node, n2::Node)::Float64 =
    sqrt(abs(n1.x - n2.x)^2 + abs(n1.y - n2.y)^2)

function distancetotal(genomes::Vector{Node})::Float64
    prev_node = genomes[1]
    distance = 0
    for i = 2:length(genomes)
        curr_node = genomes[i]
        distance += distancebetween(prev_node, curr_node)
        prev_node = curr_node
    end

    return distance
end

function makeindividual(nodes::Vector{Node})::Individual
    genomes = shuffle(nodes)
    fit = fitness(genomes)
    dist = distancetotal(genomes)

    return Individual(genomes, fit, dist)
end

function mutate(individual::Individual, mutation_rate::Float64)::Individual
    gen = copy(individual.genome)
    for i in 1:length(gen)
        if rand(Float64) < mutation_rate
            pos2 = rand(1:length(gen))
            
            tmp = gen[i]
            gen[i] = gen[pos2]
            gen[pos2] = tmp
        end
    end

    dist = distancetotal(gen)
    fit = fitness(gen)

    return Individual(gen, fit, dist)
end

fitness(genomes::Vector{Node})::Float64 =
    1 / distancetotal(genomes)

function crossover(individual1::Individual, individual2::Individual)::Individual
    (pos1, pos2) = minmax(rand(1:length(individual1.genome)), rand(1:length(individual1.genome)))

    seqm = individual1.genome[pos1:pos2]
    gen = Array{Node}(undef, length(individual2.genome)) 

    j = 1
    for i = 1:length(individual2.genome)
        if !(individual2.genome[i] in seqm)
            while j <= length(gen)
                if (j in pos1:pos2)
                    j += 1
                else
                    gen[j] = individual2.genome[i]
                    j += 1
                    break
                end
            end
        end
    end

    gen[pos1:pos2] = seqm
    fit = fitness(gen)
    dist = distancetotal(gen)

    return Individual(gen, fit, dist)
end

function selectparent(population::Vector{Individual})::Individual
    totalfitness = reduce((x, y) -> x + y.fitness, population; init=0.0)
    lottery = []
    maxpoints = 1000
    minpoints = 1

    for ind in population
        chance = ind.fitness / totalfitness
        points = Int(round(maxpoints * chance)) + minpoints
        push!(lottery, minpoints:points)

        minpoints = points
    end

    randselection = rand(1:maxpoints)

    for (i, value) in enumerate(lottery)
        if randselection in value
            return population[i]
        end
    end

    return population[rand(1:length(population))]
end

function runcycle(population::Vector{Individual}; elitism=false, mutationrate=0.05)::Vector{Individual}
    popcount = length(population)
    newpop = []

    if elitism
        push!(newpop, maximum(population))
        popcount -= 1
    end

    while popcount != 0
        p1 = selectparent(population)
        p2 = selectparent(population)

        while p1 == p2
            p2 = selectparent(population)
        end

        child = crossover(p1, p2)
        push!(newpop, child)
        popcount -= 1
    end

    map(x -> mutate(x, mutationrate), newpop)

    return newpop
end

function evolve(
    nodes::Vector{Node};
    indcount=8,
    iterations=1000,
    elitism=false,
    mutationrate=0.05)::Individual
    
    if indcount < 2
        throw(DomainError(indcount, "the number of individuals must be 2 or more"))
    end

    initpop::Vector{Individual} = []
    for _ = 1:indcount
        push!(initpop, makeindividual(nodes))
    end

    pop = initpop

    for _ = 1:iterations
        pop = runcycle(pop; elitism, mutationrate)
    end

    return maximum(pop)
end
