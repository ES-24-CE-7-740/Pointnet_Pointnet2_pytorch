#!/bin/bash
echo "Starting the LR sweep"

echo "Starting training and testing for the test with lowered LR"
sh scripts/learning_rate_tests/RUN_LR_lower.sh

echo "Starting training and testing for the test with normal LR"
sh scripts/learning_rate_tests/RUN_LR_normal.sh

echo "Starting training and testing for the test with higher LR"
sh scripts/learning_rate_tests/RUN_LR_higher.sh