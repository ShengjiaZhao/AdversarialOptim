#!/usr/bin/env bash
python gan.py --gpu=0 --z_dim=7  --repeat=$1 &
python gan.py --gpu=1 --z_dim=8 --repeat=$1 &
python gan.py --gpu=2 --z_dim=9 --repeat=$1 &
python gan.py --gpu=3 --z_dim=10 --repeat=$1 &
python gan.py --gpu=0 --z_dim=11 --repeat=$1 &
python gan.py --gpu=1 --z_dim=12 --repeat=$1 &
python gan.py --gpu=2 --z_dim=13 --repeat=$1 &
python gan.py --gpu=3 --z_dim=14 --repeat=$1 &
