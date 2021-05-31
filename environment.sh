#!/bin/sh
if test -f "~/.zshrc"; then
    source ~/.zshrc
fi
if test -f "~/.bashrc"; then
    source ~/.bashrc
fi
conda activate nbai
export PATH="$PWD/scripts:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
$SHELL