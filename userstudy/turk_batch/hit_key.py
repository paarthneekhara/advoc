import sys

hit_fp, key_fp, i = sys.argv[1:]
i = int(i)

with open(hit_fp, 'r') as f:
  l = f.read().strip().splitlines()[1:]
h = l[i].split(',')
h = [h.split('/')[-1].split('.')[0] for h in h]

with open(key_fp, 'r') as f:
  k = [l.split(',') for l in f.read().strip().splitlines()]
  uuid_to_method = {k[0]:k[1] for k in k}
  
for uuid in h:
  print(uuid_to_method[uuid])
