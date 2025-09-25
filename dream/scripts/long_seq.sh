# 1024
# Vanilla
python run.py -t=gsm8k -m=base -s=1 -l=1024 -b=32 -re 
# +DPad
python run.py -t=gsm8k -m=base -s=1 -l=1024 -b=32 -d=gaussian -k=3 -sc=2.3 -w=128 -e -re 
# +Parallel
python run.py -t=gsm8k -m=base -s=1 -l=1024 -b=32 -th=0.9 -re 
# +Parallel+DPad
python run.py -t=gsm8k -m=base -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -re 
# +Parallel+PrefixCache
python run.py -t=gsm8k -m=base -s=1 -l=1024 -b=32 -th=0.9 -c -re
# +Parallel+PrefixCache+DPad
python run.py -t=gsm8k -m=base -s=1 -l=1024 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -c -re 

# 2048
# Vanilla
python run.py -t=gsm8k -m=base -s=1 -l=2048 -b=32 -re 
# +DPad
python run.py -t=gsm8k -m=base -s=1 -l=2048 -b=32 -d=gaussian -k=3 -sc=2.3 -w=128 -e -re 
# +Parallel
python run.py -t=gsm8k -m=base -s=1 -l=2048 -b=32 -th=0.9 -re 
# +Parallel+DPad
python run.py -t=gsm8k -m=base -s=1 -l=2048 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -re 
# +Parallel+PrefixCache
python run.py -t=gsm8k -m=base -s=1 -l=2048 -b=32 -th=0.9 -c -re
# +Parallel+PrefixCache+DPad
python run.py -t=gsm8k -m=base -s=1 -l=2048 -b=32 -th=0.9 -e -d=gaussian -k=3 -sc=2.3 -w=128 -c -re 