#!/bin/bash
echo "Running entire epoch test"
echo "Starting training and testing for the test with 10 epochs"
sh scripts/epoch_tests/RUN_epoch_10.sh

echo "Starting training and testing for the test with 20 epochs"
sh scripts/epoch_tests/RUN_epoch_20.sh

echo "Starting training and testing for the test with 30 epochs"
sh scripts/epoch_tests/RUN_epoch_30.sh