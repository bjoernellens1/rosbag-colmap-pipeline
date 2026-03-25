#!/bin/bash
set -e

cd /app

if [ "$1" = "full" ]; then
    shift
    exec gttool full "$@"
elif [ "$1" = "extract" ]; then
    shift
    exec gttool extract "$@"
elif [ "$1" = "run-colmap" ]; then
    shift
    exec gttool run-colmap "$@"
elif [ "$1" = "scale-depth" ]; then
    shift
    exec gttool scale-depth "$@"
elif [ "$1" = "inspect-bag" ]; then
    shift
    exec gttool inspect-bag "$@"
else
    exec gttool "$@"
fi
