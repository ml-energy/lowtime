#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black lowtime
else
  black --check lowtime
fi

ruff check lowtime
pyright lowtime
