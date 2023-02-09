#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black rene
else
  black --check rene
fi

pydocstyle rene
pylint -j 0 rene
mypy rene
