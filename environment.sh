#!/bin/sh
if test -f "~/.zshrc"; then
    source ~/.zshrc
if test -f "~/.bashrc"; then
    source ~/.bashrc
conda activate nbai
export PATH="$PWD/scripts:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
$SHELL