"""stage.py <worktree> <base> <dir>...: copy the gitignored files under analysis/<dir>/ (no figs, no __pycache__)
whose mtime is before the base commit's time into the worktree at the same paths."""
import os,shutil,subprocess,sys
R='/home/joaquin/work/FA'; W,base,dirs=sys.argv[1],sys.argv[2],sys.argv[3:]
t=int(subprocess.check_output(['git','-C',R,'show','-s','--format=%ct',base]))
n=0
for d in dirs:
    out=subprocess.check_output(['git','-C',R,'status','--ignored','--porcelain','analysis/'+d]).decode().splitlines()
    for l in out:
        if not l.startswith('!! '): continue
        p=l[3:].rstrip('/')
        if '__pycache__' in p or '/figs' in p: continue
        src=os.path.join(R,p)
        files=[src] if os.path.isfile(src) else [os.path.join(a,f) for a,_,fs in os.walk(src) for f in fs]
        for f in files:
            if os.path.getmtime(f)<t:
                dst=os.path.join(W,os.path.relpath(f,R)); os.makedirs(os.path.dirname(dst),exist_ok=True)
                shutil.copy2(f,dst); n+=1
print(W,'staged',n)
