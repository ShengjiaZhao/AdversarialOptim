#!/usr/bin/env bash
python supervised.py --gpu=0 --z_dim=7 &
python supervised.py --gpu=1 --z_dim=8 &
python supervised.py --gpu=2 --z_dim=9 &
python supervised.py --gpu=3 --z_dim=10 &
python supervised.py --gpu=0 --z_dim=11 &
python supervised.py --gpu=1 --z_dim=12 &
python supervised.py --gpu=2 --z_dim=13 &
python supervised.py --gpu=3 --z_dim=14 &
