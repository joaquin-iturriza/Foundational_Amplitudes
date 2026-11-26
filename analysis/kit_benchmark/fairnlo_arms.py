"""Step 20 arm submissions (design PASS 2026-11-26T02:27): prints one `site submit` per arm and D.
python3 arms20.py <site> <parent_dir> <limit HH:MM:SS> [--smoke]  (--smoke: D=1e4 only, 50 steps, tags *_smoke3)"""
import sys
site, parent, limit = sys.argv[1:4]
smoke = "--smoke" in sys.argv
D = {100: 4, 316: 5, 1000: 6, 3162: 7, 10000: 8}
T = {100: 5950, 316: 8500, 1000: 8500, 3162: 8500, 10000: 8500}
BASE_FT_LR = {100: 8.11e-3, 316: 6.665e-3, 1000: 6.833e-3, 3162: 7.817e-3, 10000: 4.648e-3}
# stage-2 bests, sweeps_local/dsave_s2_{ft,scr}_ee_ttbar_nlo_d{4..8}_*/summary.txt
FT = {100: dict(ls=0.12587349935410425, ld=0.8307194216176867, wd=0.033230231315765614, reg=2.3794876716811145e-09, wu=0.12004611473530533, eta=1.0631233197500212e-10, ema="true", ed=0.9918857991909609),
      316: dict(ls=0.1224324291367089, ld=0.7588957857806236, wd=0.0013957043420621787, reg=1.9767096568530845e-08, wu=0.08882756954990328, eta=9.276144627331754e-09, ema="true", ed=0.9046396675854922),
      1000: dict(ls=7.411666559952677, ld=0.7818356410134584, wd=0.034373173098222236, reg=9.994753822951773e-10, wu=0.09270643410272897, eta=9.813685946016085e-10, ema="false", ed=0.9627266888944432),
      3162: dict(ls=0.16373774272896718, ld=0.862270564539358, wd=0.0010736412085852055, reg=5.406212926021067e-10, wu=0.18088630139827733, eta=6.679853497604928e-08, ema="true", ed=0.9025982855027542),
      10000: dict(ls=0.10903841617438866, ld=0.9997763789724559, wd=0.050752809643121836, reg=4.5173898909326057e-10, wu=0.10231071072630585, eta=3.726413546272236e-08, ema="true", ed=0.9166535316444934)}
SCR = {100: dict(lr=0.0030454378454229636, wd=0.004525496219667924, reg=1.3757094689314164e-08, wu=0.0793823047541082, eta=8.731096626941508e-08, ema="true", ed=0.9931023247875274),
       316: dict(lr=0.003999893557927694, wd=0.005785891345957641, reg=1.9536085896836016e-10, wu=0.07657384863123298, eta=1.6380637651653482e-09, ema="false", ed=0.9040982127087192),
       1000: dict(lr=0.011560180519354124, wd=0.009667655707841193, reg=7.92110947935474e-07, wu=0.19251144621521238, eta=5.424289032591735e-10, ema="false", ed=0.9969684692183509),
       3162: dict(lr=0.011235604717486412, wd=0.03435994645870153, reg=2.0916589554784696e-09, wu=0.09627581140957774, eta=8.039100499398452e-10, ema="true", ed=0.9232278547659516),
       10000: dict(lr=0.005998581665944935, wd=0.0025005085479578868, reg=9.297013286565388e-07, wu=0.05360684343613684, eta=7.90461916415821e-10, ema="false", ed=0.9288252260936424)}
common = lambda h: (f"training.scheduler=CosineAnnealingLR training.muon_weight_decay={h['wd']} training.regularization_lambda={h['reg']} "
                    f"training.cosanneal_warmup_frac={h['wu']} training.cosanneal_eta_min={h['eta']} ema={h['ema']} training.ema_decay={h['ed']}")
for d in ([10000] if smoke else D):
    t = 50 if smoke else T[d]
    sfx = "_smoke3" if smoke else ""  # _smoke, _smoke2: the 2026-11-26 failed smokes left their dirs
    f, s = FT[d], SCR[d]
    sub = f"site submit {site} FA scripts/job_kit.sh --item kit_benchmark --hdr=--time={limit} --env KIT_PARENT={parent}"
    print(f"{sub} --env KIT_LR_SCALE={f['ls']} -- kit_ee_tt__virt_finite {d} {t} A_ft training.lr={BASE_FT_LR[d]} fine_tune.layer_decay={f['ld']} fine_tune.offshell_stats=own {common(f)} tag=fairnlo_ft_d{d}{sfx}")
    print(f"{sub} -- kit_ee_tt__virt_finite {d} {t} A training.lr={s['lr']} {common(s)} tag=fairnlo_scra2_d{d}{sfx}")
    print(f"{sub} -- kit_ee_tt__virt_finite {d} {t} A training.lr={s['lr']} model.input_asym_factor=false ++data.target_over_born=true {common(s)} tag=fairnlo_vb_d{d}{sfx}")
