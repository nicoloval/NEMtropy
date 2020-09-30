#!/bin/bash
# run all tests functions in the current directory

tic=`date +%s`

for testfile in ./test_*.py
do
    echo $testfile
    python3 $testfile
done

toc=`date +%s`
runtime=$((toc-tic))
echo runtime "$runtime"

