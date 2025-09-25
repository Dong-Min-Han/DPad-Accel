

# Vanilla
python llada/run.py -t=gsm8k-split -m=1.5 -s=1 -l=1024 -b=32 -re 

# +DPad
python llada/run.py -t=gsm8k-split -m=1.5 -s=1 -l=1024 -b=32 -d=gaussian -k=3 -sc=1.6 -w=256 -e -re 

# +Parallel
python llada/run.py -t=gsm8k-split -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -re 

# +Parallel+DPad
python llada/run.py -t=gsm8k-split -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=1.6 -w=256 -re 

# +Parallel+PrefixCache
python llada/run.py -t=gsm8k-split -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -c -re

# +Parallel+PrefixCache+DPad
python llada/run.py -t=gsm8k-split -m=1.5 -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=1.6 -w=256 -c -re 