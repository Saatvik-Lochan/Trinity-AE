#!/bin/bash

echo "**********************[LLaMA Vanilla]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t vanilla --n 946 --baseline flashinfer
    sleep 1
done
echo "**********************[LLaMA PreNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t prenorm --n 579 --baseline flashinfer
    sleep 1
done
echo "**********************[LLaMA QKNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m llama --t qknorm --n 2321 --baseline flashinfer
    sleep 1
done

echo "**********************[Falcon Vanilla]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t vanilla --n 1562 --baseline flashinfer
    sleep 1
done
echo "**********************[Falcon PreNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t prenorm --n 579 --baseline flashinfer
    sleep 1
done
echo "**********************[Falcon QKNorm]****************************"
for i in {1..3}; do
    echo "<Trial $i>"
    python run_eval.py --o 1 --m falcon --t qknorm --n 3690 --baseline flashinfer
    sleep 1
done