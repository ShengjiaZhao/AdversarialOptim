#!/usr/bin/env bash
python3 supervised.py --gpu=0 --z_dim=7 &
python3 supervised.py --gpu=1 --z_dim=8 &
python3 supervised.py --gpu=2 --z_dim=9 &
python3 supervised.py --gpu=3 --z_dim=10 &
python3 supervised.py --gpu=0 --z_dim=11 &
python3 supervised.py --gpu=1 --z_dim=12 &
python3 supervised.py --gpu=2 --z_dim=13 &
python3 supervised.py --gpu=3 --z_dim=14 &
