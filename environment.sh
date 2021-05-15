#!/bin/sh
source ~/.zshrc
conda activate nbai
export PATH="$PWD/scripts:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
$SHELL