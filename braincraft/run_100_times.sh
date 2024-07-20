#!/bin/bash

echo -e "Neural network testing...\n"
for i in {1..100}
do
    echo -e "Starting program $i times:"
    ./program
    sleep 1
done
