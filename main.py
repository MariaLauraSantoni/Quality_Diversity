import numpy as np
from pyDOE2 import lhs
from ioh import get_problem, ProblemClass


def quality_diversity(function, instance, dimension, initial_size, size_best, lb, ub, iterations):
    # Definisci il numero di campioni e la dimensione dello spazio
    # n_samples = 100  # Numero di campioni
    # n_dimensions = 3  # Dimensione dello spazio
    # Definisci i limiti inferiori e superiori per ciascuna dimensione
    lower_bound =  [lb]* dimension # Esempio di limiti inferiori
    upper_bound = [ub]* dimension   # Esempio di limiti superiori
    # Genera il campione LHS
    lhs_sample = lhs(dimension, samples=initial_size, criterion='maximin')
    # Scala il campione LHS nei limiti specificati
    scaled_lhs_sample = np.zeros_like(lhs_sample)

    for i in range(dimension):
        scaled_lhs_sample[:, i] = lower_bound[i] + lhs_sample[:, i] * (upper_bound[i] - lower_bound[i])
    # Stampare il campione generato
    print("Campione LHS generato:")
    print(scaled_lhs_sample)

    # Evaluate f on each point of the sample
    f = get_problem(function, instance, dimension, ProblemClass.BBOB)
    values = np.array([])
    for i in scaled_lhs_sample:
        values = np.append(values, f(i))
    # Ottieni gli indici per il riordino
    ind= np.argsort(values)
    # Applica il riordino agli array di punti e punti legati
    values_ord = values[ind]
    sample_ord = scaled_lhs_sample[ind]
    loss = values_ord - f.optimum.y
    # Calcolo average value of the function loss tra i primi 20 valori migliori LEI LA VOGLIO SALVARE IN UN FILE OGNI ITERAZIONE
    mean_loss = np.mean(loss[:size_best])
    print(mean_loss)

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
        mean_dist = np.mean(min_dist)
        #QUI DEVO APRIRE IL FILE E STAMPARE TXT I DUE VALORI
        with open(str(function) + '_' + str(instance) + '_'  + str(dimension)+ '_' + str(initial_size) + '_' + str(size_best) + '_' + str(lb) + '_' + str(ub)+ '_' + str(iterations)+ '.txt', 'a') as file:
            # Scrivi i valori separati da uno spazio sulla stessa riga
            line = f'{mean_dist} {mean_loss}\n'  # Usiamo '\n' per andare a capo dopo ogni riga
            file.write(line)
        #ORA E' IL MOMENTO DELLO SWAP
        loss_store = np.array([])
        loss_swap = np.array([])
        new_array = []
        for j in range(len(index)):
            ind_swap = index[j]
            for i in range(size_best, len(sample_ord)):
                # Creare una copia dell'array iniziale per effettuare lo scambio
                array_new = np.copy(sample_ord)

                # Effettuare lo scambio tra il numero in posizione 4 e il numero in posizione i
                array_new[ind_swap], array_new[i] = array_new[i], array_new[i]

                min_dist_swap = np.zeros((size_best,))
                min_index_swap = np.zeros((size_best,))
                # Itera su tutti i punti
                # Calcola la differenza tra il punto swap e tutti gli altri punti
                diff_swap = array_new[:size_best] - array_new[:size_best][ind_swap, :]  # Trasforma in un vettore delle differenze
                dist_swap = np.linalg.norm(diff_swap, axis=1)  # Calcola la norma (distanza euclidea) per ciascuna riga
                dist_swap[ind_swap] = np.inf  # Imposta la distanza a infinito per il punto stesso
                # Trova la minima distanza per il punto i
                min_dist_one_swap = np.min(dist_swap)
                if min_dist_one_swap > min_dist_one:
                    verified_cond[j] = True
                    loss_new = np.copy(loss)
                    loss_new[ind_swap] = f(array_new[i]) - f.optimum.y
                    loss_swap = np.append(loss_swap, loss_new[ind_swap])
                    loss1= np.mean(loss_new[:size_best])
                    loss_store=np.append(loss_store, loss1)
                    index_to_swap = i

                    # Aggiungere l'array modificato alla lista
                    new_array.append(array_new)
                    break
        if not all(verified_cond):
            print(f"Could not find a point that satisfies the distance criterion during iteration {z}")
            break  # Passa all'iterazione successiva del ciclo esterno



        ind_loss = np.argmin(loss_store)
        loss = np.copy(loss)
        a = loss[index[ind_loss]]
        loss[index[ind_loss]] = loss_swap[ind_loss]
        loss[index_to_swap] = a
        mean_loss=np.mean(loss[:size_best])
        sample_ord= new_array[ind_loss]




if __name__ == "__main__":
    # inp1 = int(input("function: "))
    # inp2 = int(input("instance: "))
    # inp3 = int(input("dimension: "))
    # inp4 = int(input("initial_size: "))
    # inp5 = int(input("size_best: "))
    # inp6 = int(input("lb: "))
    # inp7 = int(input("ub: "))
    # inp8 = int(input("iterations: "))

    inp1 = 3
    inp2 = 0
    inp3 = 2
    inp4 = 1000
    inp5 = 20
    inp6 = -5
    inp7 = 5
    inp8 = 1000

    quality_diversity(inp1,inp2,inp3,inp4,inp5,inp6,inp7,inp8)