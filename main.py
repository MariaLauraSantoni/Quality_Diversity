import numpy as np
from pyDOE2 import lhs
from ioh import get_problem, ProblemClass
import sys

def quality_diversity(function, instance, dimension, initial_size, size_best, lb, ub, iterations):
    # Definisci il numero di campioni e la dimensione dello spazio
    # n_samples = 100  # Numero di campioni
    # n_dimensions = 3  # Dimensione dello spazio
    # Definisci i limiti inferiori e superiori per ciascuna dimensione
    lower_bound = [lb]* dimension # Esempio di limiti inferiori
    upper_bound = [ub]* dimension   # Esempio di limiti superiori
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
    ind= np.argsort(values)
    # Applica il riordino agli array di punti e punti legati
    values_ord = values[ind]
    sample_ord = random_sample[ind]
    loss = values_ord - f.optimum.y
    # Calcolo average value of the function loss tra i primi 20 valori migliori LEI LA VOGLIO SALVARE IN UN FILE OGNI ITERAZIONE
    mean_loss = np.mean(loss[:size_best])
    #print(mean_loss)

    # Supponiamo che 'punti' sia una matrice NumPy con ogni riga rappresentante un punto n-dimensionale.
    # Ad esempio, punti.shape sarà (numero_di_punti, n_dimensioni).

    # numero_di_punti, n_dimensioni = punti.shape
    # Creiamo una matrice vuota per memorizzare le distanze minime tra i punti.
    for z in range(iterations):
        verified_cond = [False ,False]  # Variabile di controllo
        min_dist = np.zeros((size_best,))
        # Itera su tutti i punti
        for i in range(size_best):
            # Calcola la differenza tra il punto i e tutti gli altri punti
            diff = sample_ord[:size_best] - sample_ord[:size_best][i, :]  # Trasforma in un vettore delle differenze
            dist = np.linalg.norm(diff, axis=1)  # Calcola la norma (distanza euclidea) per ciascuna riga
            dist[i] = np.inf  # Imposta la distanza a infinito per il punto stesso
            # Trova la minima distanza per il punto i
            min_dist_one = np.min(dist)
            # Memorizza la minima distanza nel vettore delle distanze minime
            min_dist[i] = min_dist_one
        #TROVO LA DISTANZA MINIMA TRA LE DISTANZE MINIME E LA COPPIA CHE MI RESTITUISCE LA DISTANZA MINIMA
        total_min_dist = np.min(min_dist)
        index = np.where(min_dist == total_min_dist)[0]
        #ALTRA COSA CHE VOGLIO STAMPARE POTREBBE ESSERE LEI LA MEDIA DEI VALORI IN MIN_DIST.
        #mean_dist = np.mean(min_dist)
        #QUI DEVO APRIRE IL FILE E STAMPARE TXT I DUE VALORI
        with open(str(function) + '_' + str(instance) + '_'  + str(dimension)+ '_' + str(initial_size) + '_' + str(size_best) + '_' + str(lb) + '_' + str(ub)+ '_' + str(iterations)+ '.txt', 'a') as file:
            # Scrivi i valori separati da uno spazio sulla stessa riga
            line = f'{total_min_dist} {mean_loss}\n'  # Usiamo '\n' per andare a capo dopo ogni riga
            file.write(line)
        #ORA E' IL MOMENTO DELLO SWAP
        loss_store = np.array([])
        loss_swap = np.array([])
        new_array = []
        index_to_swap = []
        for j in range(len(index)):
            ind_swap = index[j]
            for i in range(size_best, len(sample_ord)):
                # Creare una copia dell'array iniziale per effettuare lo scambio
                array_new = np.copy(sample_ord)
                array_new = np.copy(sample_ord)
                temp = sample_ord[ind_swap]
                print(temp)
                # Assegna il valore dell'altro elemento al primo
                array_new[ind_swap] = array_new[i]
                print(temp)
                # Assegna il valore temporaneo al secondo elemento
                array_new[i] = temp

                min_dist_swap = np.zeros((size_best,))
                min_index_swap = np.zeros((size_best,))
                # Itera su tutti i punti
                # Calcola la differenza tra il punto swap e tutti gli altri punti
                diff_swap = array_new[:size_best] - array_new[:size_best][ind_swap, :]  # Trasforma in un vettore delle differenze
                dist_swap = np.linalg.norm(diff_swap, axis=1)  # Calcola la norma (distanza euclidea) per ciascuna riga
                dist_swap[ind_swap] = np.inf  # Imposta la distanza a infinito per il punto stesso
                # Trova la minima distanza per il punto i
                min_dist_one_swap = np.min(dist_swap)
                if min_dist_one_swap > total_min_dist:
                    verified_cond[j] = True
                    loss_new = np.copy(loss)
                    loss_new[ind_swap] = f(array_new[ind_swap]) - f.optimum.y
                    loss_swap = np.append(loss_swap, loss_new[ind_swap])
                    loss1= np.mean(loss_new[:size_best])
                    loss_store=np.append(loss_store, loss1)
                    index_to_swap.append(i)

                    # Aggiungere l'array modificato alla lista
                    new_array.append(array_new)
                    break

        if all(element is False for element in verified_cond):
            print(f"Could not find a point that satisfies the distance criterion during iteration {z}")
            break  # Passa all'iterazione successiva del ciclo esterno
        elif all(element is True for element in verified_cond):
            ind_loss = np.argmin(loss_store)
        elif verified_cond.count(True) == 1:
            ind_loss = verified_cond.index(True)

        loss = np.copy(loss)
        a = loss[index[ind_loss]]
        loss[index[ind_loss]] = loss_swap[ind_loss]
        loss[index_to_swap] = a
        mean_loss = np.mean(loss[:size_best])
        sample_ord = new_array[ind_loss]




        # loss = np.copy(loss)
        # a = loss[index[ind_loss]]
        # loss[index[ind_loss]] = loss_swap[ind_loss]
        # loss[index_to_swap] = a
        # mean_loss=np.mean(loss[:size_best])
        # sample_ord= new_array[ind_loss]




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
   inp5 = 20
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
   quality_diversity(inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8)
