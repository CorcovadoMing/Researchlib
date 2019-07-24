#!/bin/sh

pushd tests
ln -s ../researchlib .
pytest test* -v
rm researchlib
popd
