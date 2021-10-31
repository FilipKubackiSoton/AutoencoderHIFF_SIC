import numpy as np 
import math
import copy

def rand_bin_array(K, N):
    """
    THe function return random binary string. 
    The string has K - 0's adn N-K - 1's
    """
    arr = np.zeros(N)
    arr[:K]  = 1
    np.random.shuffle(arr)
    return arr

def hiff_fitness(array):
    """
    Calculate and return value related to h-iff 
    assignment to the binary string of array. 
    """

    def f(val):
        if val == 1 or val == 0:
            return 1
        else:
            return 0

    def t(left, right):
        if left == 1 and right == 1:
            return 1
        elif left == 0 and right == 0:
            return 0
        else:
            return None 

    def val_recursive(array, flor, sum):
        if flor > levels:
            return sum
        arr = []
        power = 2 ** flor
        for i in range(0,2**(levels - flor)-1,2):
            arr.append(t(array[i], array[i+1]))
            sum = sum + (f(array[i]) + f(array[i+1]))* power
        return val_recursive(arr, flor + 1, sum)

    size = len(array)
    if not (size/2).is_integer():
        raise ValueError("Array size must be power of 2.")
    levels = int(math.log2(size))
    sum = 0
    return val_recursive(array, 0,  sum)
        

def generate_training_sat(N, set_size):
    """
    Generate training set for H-IFF problem. 
    
    return: binary array of size N to train NN
    """
    input = np.ndarray(shape=(set_size, N))
    output = np.ndarray(shape=(set_size, N))

    if not (math.log2(N)).is_integer():
            raise ValueError("Array size must be power of 2.")
    for k in range(set_size):
        candidate_solution = np.random.randint(2, size = N)
        input[k]=candidate_solution
        solution_fitness = hiff_fitness(candidate_solution)
        for i in range(10 * N):
            index = np.random.randint(N)
            new_candidate_sol = copy.copy(candidate_solution)
            new_candidate_sol[index] = 1 - new_candidate_sol[index] # apply variation 
            new_fitness = hiff_fitness(new_candidate_sol) # check the change 
            if new_fitness >= solution_fitness : 
                candidate_solution = new_candidate_sol
                solution_fitness = new_fitness
        output[k]=candidate_solution

    return input, output