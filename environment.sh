#!/bin/sh
if test -f "~/.zshrc"; then
    source ~/.zshrc
fi
if test -f "~/.bashrc"; then
    source ~/.bashrc
fi
export PATH="$PWD/scripts:$PATH"
export PYTHONPATH="$PWD:$PYTHONPATH"
$SHELL