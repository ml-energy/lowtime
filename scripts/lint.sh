#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  black poise
else
  black --check poise
fi

ruff poise
