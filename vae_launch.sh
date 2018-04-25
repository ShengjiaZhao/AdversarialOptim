#!/usr/bin/env bash

python vae.py --gpu=0 --z_dim=7 --beta=$1 &
python vae.py --gpu=1 --z_dim=8 --beta=$1 &
python vae.py --gpu=2 --z_dim=9 --beta=$1 &
python vae.py --gpu=3 --z_dim=10 --beta=$1 &
python vae.py --gpu=0 --z_dim=11 --beta=$1 &
python vae.py --gpu=1 --z_dim=12 --beta=$1 &
python vae.py --gpu=2 --z_dim=13 --beta=$1 &
python vae.py --gpu=3 --z_dim=14 --beta=$1 &