#!/bin/bash

#SBATCH --job-name=ciao
#SBATCH --array=0-##jobs_count##
#SBATCH --clusters=serial
#SBATCH --partition=serial_std
#SBATCH --mem=2048MB
#SBATCH --time=96:00:00
#SBATCH --mail-user=marialaura.santoni@studenti.unicam.it
#SBATCH --mail-type=END,FAIL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=boh.out
#SBATCH --err=boh.err


import sys

for run in {1..10}; do
    # Crea una cartella per ogni esecuzione
    mkdir run_$run

    # Definisci gli argomenti per lo script Python
    arg1="5"
    arg2="0"
    arg3="10"
    arg4="1000"
    arg5="20"
    arg6="-5"
    arg7="5"
    arg8="10"
   


   # Esegui lo script Python con gli argomenti
   python main.py "$arg1" "$arg2" "$arg3" "$arg4" "$arg5" "$arg6" "$arg7" "$arg8"
   mv $arg1* run_$run

done
