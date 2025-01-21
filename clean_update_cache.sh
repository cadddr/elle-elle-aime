#!/bin/bash
cd cache;
git clean -f .;
cd ..;
git submodule update;
