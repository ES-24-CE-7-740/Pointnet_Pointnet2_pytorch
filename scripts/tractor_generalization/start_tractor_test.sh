#!/bin/bash
echo "Running entire tractor zeroshot test"
echo "Starting training and testing for the blue valtra tractor"
sh scripts/tractor_generalization/RUN_blue_valtra.sh

echo "Starting training and testing for the grey valtra tractor"
sh scripts/tractor_generalization/RUN_grey_valtra.sh

echo "Starting training and testing for the fendt300 tractor"
sh scripts/tractor_generalization/RUN_fendt300.sh

echo "Starting training and testing for the fendt1000 tractor"
sh scripts/tractor_generalization/RUN_fendt1000.sh

echo "Starting training and testing for the orange valtra tractor"
sh scripts/tractor_generalization/RUN_orange_valtra.sh

echo "Starting training and testing for the red valtra tractor"
sh scripts/tractor_generalization/RUN_red_valtra.sh