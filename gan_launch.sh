#!/usr/bin/env bash
python gan.py --gpu=0 --z_dim=7 &
python gan.py --gpu=1 --z_dim=8 &
python gan.py --gpu=2 --z_dim=9 &
python gan.py --gpu=3 --z_dim=10 &
python gan.py --gpu=0 --z_dim=11 &
python gan.py --gpu=1 --z_dim=12 &
python gan.py --gpu=2 --z_dim=13 &
python gan.py --gpu=3 --z_dim=14 &
