#!/bin/bash
# run all tests functions in the current directory

for testfile in ./test_*.py
do
    echo $testfile
    python3 $testfile
done

