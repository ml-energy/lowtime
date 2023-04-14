#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black rene
else
  black --check rene
fi

ruff rene
mypy rene
