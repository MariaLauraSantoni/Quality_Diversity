import time

import numpy as np
#from pyDOE2 import lhs
from ioh import get_problem, ProblemClass
import sys

def quality_diversity(function, instance, dimension, initial_size, size_best, lb, ub, iterations):
    # Definisci il numero di campioni e la dimensione dello spazio
    # n_samples = 100  # Numero di campioni
    # n_dimensions = 3  # Dimensione dello spazio
    # Definisci i limiti inferiori e superiori per ciascuna dimensione
    lower_bound = [lb] * dimension  # Esempio di limiti inferiori
    upper_bound = [ub] * dimension  # Esempio di limiti superiori
    # Genera il campione LHS
    # lhs_sample = lhs(dimension, samples=initial_size, criterion='maximin')
    # # Scala il campione LHS nei limiti specificati
    # random_sample = np.zeros_like(lhs_sample)

    # for i in range(dimension):
    #     random_sample[:, i] = lower_bound[i] + lhs_sample[:, i] * (upper_bound[i] - lower_bound[i])
    # # Stampare il campione generato
    # #print("Campione LHS generato:")
    # #print(random_sample)

    # Crea un campione di punti casuali
    random_sample = np.random.uniform(low=lower_bound, high=upper_bound, size=(initial_size, dimension))

    # Evaluate f on each point of the sample
    f = get_problem(function, instance, dimension, ProblemClass.BBOB)
    values = np.array([])
    for i in random_sample:
        values = np.append(values, f(i))
    # Ottieni gli indici per il riordino
    before_sort = time.process_time()
    #print(before_sort)
    ind = np.argsort(values)
    # Applica il riordino agli array di punti e punti legati
    values_ord = values[ind]
    sample_ord = random_sample[ind]
    # Specifica il percorso del file in cui desideri scrivere i dati
    file_path = str(function) + '_' + str(instance) + '_' + str(dimension) + '_' + str(initial_size) + '_' + str(
            size_best) + '_' + str(lb) + '_' + str(ub) + '_' + str(iterations) + '_sample_values.txt'

    # Usa numpy.savetxt per scrivere i dati in due colonne separate
    np.savetxt(file_path, np.column_stack((values_ord, sample_ord)), delimiter='\t', header="f(x)\tx", comments='')
    # Apre un file in modalità scrittura
    with open('distances.txt', 'w') as file:
        # Calcola la distanza euclidea tra ogni coppia di punti e scrivi sul file
        for i in range(10000):
            for j in range(10000):
                distance = np.sqrt(np.sum((sample_ord[i] - sample_ord[j]) ** 2))
                file.write(f"{distance}\t")
            file.write("\n")
    # Opzionalmente, puoi specificare il delimiter per separare le colonne (nel caso sopra, ho usato il tab '\t')
    # e un'intestazione per le colonne. I commenti sono impostati su '' per evitare commenti nel file.
    # with open(str(function) + '_' + str(instance) + '_' + str(dimension) + '_' + str(initial_size) + '_' + str(
    #         size_best) + '_' + str(lb) + '_' + str(ub) + '_' + str(iterations) + '_sample_values.txt', 'a') as file:
    #     # Scrivi i valori separati da uno spazio sulla stessa riga
    #     line = f'{values_ord} {sample_ord}\n'  # Usiamo '\n' per andare a capo dopo ogni riga
    #     file.write(line)
    sort= time.process_time()
    #print(sort)
    loss = values_ord - f.optimum.y
    # Calcolo average value of the function loss tra i primi 20 valori migliori LEI LA VOGLIO SALVARE IN UN FILE OGNI ITERAZIONE
    mean_loss = np.mean(loss[:size_best])
    # print(mean_loss)


    # Supponiamo che 'punti' sia una matrice NumPy con ogni riga rappresentante un punto n-dimensionale.
    # Ad esempio, punti.shape sarà (numero_di_punti, n_dimensioni).

    # numero_di_punti, n_dimensioni = punti.shape
    # Creiamo una matrice vuota per memorizzare le distanze minime tra i punti.
    min_dist = np.full((size_best, size_best), np.inf)
    #min_dist1 = np.zeros((size_best,))
    for i in range(size_best):
        # Calcola la differenza tra il punto i e tutti gli altri punti
        diff = sample_ord[:size_best] - sample_ord[:size_best][i, :]  # Trasforma in un vettore delle differenze
        dist = np.linalg.norm(diff, axis=1)  # Calcola la norma (distanza euclidea) per ciascuna riga
        dist[i] = np.inf  # Imposta la distanza a infinito per il punto stesso
        min_dist[i, :] = dist
        # Trova la minima distanza per il punto i
        #min_dist_one = np.min(dist)
        # # Memorizza la minima distanza nel vettore delle distanze minime
        #min_dist1[i] = min_dist_one
    # TROVO LA DISTANZA MINIMA TRA LE DISTANZE MINIME E LA COPPIA CHE MI RESTITUISCE LA DISTANZA MINIMA
    total_min_dist = np.min(min_dist)
    #total_min_dist1 = np.min(min_dist1)
    # Trova gli indici della prima occorrenza del valore minimo nella matrice
    index = np.argmin(min_dist)
    index = np.unravel_index(index, min_dist.shape)
    #index1 = np.where(min_dist == total_min_dist)[0]
    #print(index)
    #Beginning of the swap loop
    for z in range(iterations):
        verified_cond = [False ,False]
        with open(str(function) + '_' + str(instance) + '_'  + str(dimension)+ '_' + str(initial_size) + '_' + str(size_best) + '_' + str(lb) + '_' + str(ub)+ '_' + str(iterations)+ '.txt', 'a') as file:
            # Scrivi i valori separati da uno spazio sulla stessa riga
            line = f'{total_min_dist} {mean_loss}\n'  # Usiamo '\n' per andare a capo dopo ogni riga
            file.write(line)

        ###ORA E' IL MOMENTO DELLO SWAP
        diff_store = []
        new_array = []
        dist_store = []
        index_to_swap = []
        for j in range(len(index)):
            ind_swap = index[j]
            for i in range(size_best, len(sample_ord)):
                # Creare una copia dell'array iniziale per effettuare lo scambio
                array_new = np.copy(sample_ord)
                temp = sample_ord[ind_swap]
                #print(temp)
                # Assegna il valore dell'altro elemento al primo
                array_new[ind_swap] = array_new[i]
                #print(temp)
                # Assegna il valore temporaneo al secondo elemento
                array_new[i] = temp
                # Effettuare lo scambio tra il numero in posizione 4 e il numero in posizione i
                #array_new[ind_swap], array_new[i] = array_new[i], array_new[ind_swap]
                min_dist_swap = np.zeros((size_best,))
                min_index_swap = np.zeros((size_best,))
                # Itera su tutti i punti
                # Calcola la differenza tra il punto swap e tutti gli altri punti
                diff_swap = array_new[:size_best] - array_new[:size_best][ind_swap,:]  # Trasforma in un vettore delle differenze
                dist_swap = np.linalg.norm(diff_swap, axis=1)  # Calcola la norma (distanza euclidea) per ciascuna riga
                dist_swap[ind_swap] = np.inf  # Imposta la distanza a infinito per il punto stesso
                #per poter sostituire solo la riga giusta alla matrice senza dover ricalcolare tutte le distanze
                # Trova la minima distanza per il punto i
                min_dist_one_swap = np.min(dist_swap)
                if min_dist_one_swap > total_min_dist:
                    #print(total_min_dist)
                    verified_cond[j] = True
                    #ora sono scambiati quindi sto facendo quello scambiato - quello vecchio della coppia migliore
                    dist_store.append(dist_swap)
                    diff_store.append(f(array_new[ind_swap])-f(array_new[i]))
                    #diff_store.append(f(array_new[ind_swap]))
                    # Aggiungere l'array modificato alla lista
                    new_array.append(array_new)
                    index_to_swap.append(i)
                    #print(i)
                    break

        if all(element is False for element in verified_cond):
            print(f"Could not find a point that satisfies the distance criterion during iteration {z}")
            break  # Passa all'iterazione successiva del ciclo esterno
        elif all(element is True for element in verified_cond):
            ind_loss = np.argmin(diff_store)
            sample_ord = new_array[ind_loss]
            # new_loss = f(sample_ord[index[ind_loss]]) - f.optimum.y
            mean_loss = (mean_loss * size_best + diff_store[ind_loss]) / size_best
            # min_dist_new = np.copy(min_dist)
            min_dist[index[ind_loss], :] = dist_store[ind_loss]
            min_dist[:, index[ind_loss]] = dist_store[ind_loss]
            total_min_dist = np.min(min_dist)
            # Trova gli indici della prima occorrenza del valore minimo nella matrice
            index = np.argmin(min_dist)
            index = np.unravel_index(index, min_dist.shape)
        elif verified_cond.count(True) == 1:
            ind_loss = verified_cond.index(True)
            sample_ord = new_array[0]
            # new_loss = f(sample_ord[index[ind_loss]]) - f.optimum.y
            mean_loss = (mean_loss * size_best + diff_store[0]) / size_best
            # min_dist_new = np.copy(min_dist)
            min_dist[index[ind_loss], :] = dist_store[0]
            min_dist[:, index[ind_loss]] = dist_store[0]
            total_min_dist = np.min(min_dist)
            # Trova gli indici della prima occorrenza del valore minimo nella matrice
            index = np.argmin(min_dist)
            index = np.unravel_index(index, min_dist.shape)






                #DEVO RICALCOLARE TOTAL_MIN_DIST AD UNA CERTA RICORDATELO!!
if __name__ == "__main__":
    # inp1 = int(input("function: "))
    # inp2 = int(input("instance: "))
    # inp3 = int(input("dimension: "))
    # inp4 = int(input("initial_size: "))
    # inp5 = int(input("size_best: "))
    # inp6 = int(input("lb: "))
    # inp7 = int(input("ub: "))
    # inp8 = int(input("iterations: "))

    # inp1 = sys.argv[1]
    # inp2 = sys.argv[2]
    # inp3 = sys.argv[3]
    # inp4 = sys.argv[4]
    # inp5 = sys.argv[5]
    # inp6 = sys.argv[6]
    # inp7 = sys.argv[7]
    # inp8 = sys.argv[8]

    inp1 = 5
    inp2 = 0
    inp3 = 2
    inp4 = 10000
    inp5 = 5
    inp6 = -5
    inp7 = 5
    inp8 = 1000

    inp1 = int(inp1)
    inp2 = int(inp2)
    inp3 = int(inp3)
    inp4 = int(inp4)
    inp5 = int(inp5)
    inp6 = int(inp6)
    inp7 = int(inp7)
    inp8 = int(inp8)
    quality_diversity(inp1, inp2, inp3, inp4, inp5, inp6, inp7, inp8)