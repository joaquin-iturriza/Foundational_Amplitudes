import json,sys,math
gm=lambda xs: math.exp(sum(map(math.log,xs))/len(xs))
recs=[]
for f in sys.argv[1:]:
    recs+=[json.loads(l[5:]) for l in open(f) if l.startswith('EVAL ')]
fin=recs[0]['val_mse_prepd']
NEW=('udbar_Wgggg','udbar_Wggggg')
old=[k for k in fin if k not in NEW and k!='combined']
f0=gm([fin[k] for k in old])
print(len(old),'finale gm %.3g'%f0)
for r in recs:
    v=r['val_mse_prepd']; o=[v[k] for k in old]; rat=[v[k]/fin[k] for k in old]
    print(r['weights'].split('/')[1:3], '%.3g'%gm(o), '%.3g'%(gm(o)/f0), '%.3g-%.3g'%(min(rat),max(rat)), sum(x>1.1 for x in rat), max(zip(rat,old))[1])
