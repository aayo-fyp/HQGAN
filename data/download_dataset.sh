#!/bin/bash
curl -L -o gdb9.tar.gz http://deepchem.io.s3-website-us-west-1.amazonaws.com/datasets/gdb9.tar.gz
tar xvzf gdb9.tar.gz
rm gdb9.tar.gz

curl -L -o NP_score.pkl.gz https://github.com/gablg1/ORGAN/raw/master/organ/NP_score.pkl.gz
curl -L -o SA_score.pkl.gz https://github.com/gablg1/ORGAN/raw/master/organ/SA_score.pkl.gz
