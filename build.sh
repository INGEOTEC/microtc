#!/bin/bash

cd $RECIPE_DIR
# echo "Building !!!!" `pwd` $RECIPE_DIR
$PYTHON setup.py install --single-version-externally-managed --record=record.txt
