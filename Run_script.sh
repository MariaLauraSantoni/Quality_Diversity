#!/bin/bash

#SBATCH --job-name=f1
#SBATCH --array=0-##jobs_count##
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --mem=2048MB
#SBATCH --time=96:00:00
#SBATCH --mail-user=marialaura.santoni@studenti.unicam.it
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=script.out
#SBATCH --err=script.err


import sys

function="1"

mkdir results_$function
for run in {1..10}; do
    # Crea una cartella per ogni esecuzione

    # Definisci gli argomenti per lo script Python
    #function="1"
    instance="0"
    dimension="2"
    initial_size="10000"
    best_size="5"
    lb="-5"
    ub="5"
    iterations="1000"
    cd results_$function
    mkdir run_$run
    cd ..

   # Esegui lo script Python con gli argomenti
   python main.py "$function" "$instance" "$dimension" "$initial_size" "$best_size" "$lb" "$ub" "$iterations"
   mv $function* results_$function/run_$run_$function

done
