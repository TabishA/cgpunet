import os


def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise NotADirectoryError(path)


def get_dominant_solution(p, q):
    obj = len(p)
    dominance = []
    for i in range(obj):
        if p[i] >= q[i]:
            dominance.append(True)
        else:
            dominance.append(False)

    if True in dominance and False not in dominance:
        return p
    elif True not in dominance and False in dominance:
        return q
    else:
        return None


def identify_pareto(population):
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    fitness = {}
    for i in range(len(population)):
        ft = []
        ft.append(population[i].eval)
        ft.append(population[i].novelty)
        fitness[i] = ft


    ranked_solutions = {}
    frontal = set()
    for key, current_solution in fitness.items():
        ranked_solutions[key] = {'Sp': set(), 'Np': 0}

        for index, solution in fitness.items():
            if current_solution[0:2] != solution[0:2]:
                dominant = get_dominant_solution(current_solution[0:2], solution[0:2])
                if dominant is None:
                    continue
                if dominant == current_solution[0:2]:
                    ranked_solutions[key]['Sp'].add(index)
                elif dominant == solution[0:2]:
                    ranked_solutions[key]['Np'] += 1

        if ranked_solutions[key]['Np'] == 0:
            ranked_solutions[key]['Rank'] = 1
            fitness[key].append(1)
            frontal.add(key)

    i = 1
    while len(frontal) != 0:
        sub = set()
        for sol in frontal:
            for dominated_solution in ranked_solutions[sol]['Sp']:
                ranked_solutions[dominated_solution]['Np'] -= 1
                if ranked_solutions[dominated_solution]['Np'] == 0:
                    ranked_solutions[dominated_solution]['Rank'] = i + 1
                    fitness[dominated_solution].append(i + 1)
                    sub.add(dominated_solution)
        i += 1
        frontal = sub

    rank1_solutions = []
    for key, value in fitness.items():
        if value[2] == 1:
            rank1_solutions.append(population[key])

    return rank1_solutions
