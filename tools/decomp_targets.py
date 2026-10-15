#!/usr/bin/env python
"""Decomposed targets (D48): |M|^2 split into the parts a real calculation sums.

A tree amplitude in MadGraph's standalone_cpp is built per helicity h as
    amp_a(h)          one entry per Feynman-diagram amplitude (a = 0..n_amp-1),
    J_i(h) = sum_a C_ia amp_a(h)     the colour-ordered (colour-flow) partial amplitudes,
    |M|^2 = sum_h sum_ij G_ij Re(J_i(h) J_j(h)^*)      G = colour matrix / (denom * spin-colour average).
Two bases follow, and both are written:
  colour  : X^c_ij = sum_h J_i J_j^*   (n_col x n_col),  |M|^2 = sum_ij G_ij Re X^c_ij.
            The colour-ordered partial amplitudes of real QCD calculations: gauge invariant one by one.
  diagram : X^d_ab = sum_h amp_a amp_b^*  (n_amp x n_amp), |M|^2 = Re sum_ab K_ab X^d_ab,  K = C^T G C^*.
            Diagram by diagram (MadGraph's gauge: unitary for the electroweak bosons, Feynman for the gluon);
            the only decomposition left when the process has a single colour flow.
X is the helicity-summed Gram matrix of the parts: the overall phase of each helicity amplitude is a
convention and cancels in it, so X is the phase-free target. Its diagonal is |part|^2, its off-diagonal the
interference of two parts.

The patch is textual and small: CPPProcess.cc copies the colour-flow vector J out of matrix_1_* and
accumulates both Gram matrices after every helicity in sigmaKin; dc_coeffs() replays MadGraph's own
jamp block on unit amplitudes to read C. Nothing is re-derived: G, C and the averaging come from the
generated code, and every event's coherent sum is checked against MadGraph's own matrix_element and,
for a pool, against the stored |M|^2.

The process's production standalone (<MG5_WORK_DIR>/<proc>_standalone, the param card the pool was
labelled with) is copied to <workdir>/<proc>_decomp_sa and patched there; the production copy is
never touched. alpha_s is set per event at mu = sqrt(s), as the pool's labelling does.

Usage:
    python tools/decomp_targets.py --process uubar_gg --pool train.npy --out train.decomp.npz
    python tools/decomp_targets.py --process uubar_gg --recipe recipes/transfer_probe_uubar_gg.yaml \\
        --out-dir $DATA_DIR/decomp            # every role's pool of the recipe, as training reads them
"""
import argparse
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import mg5_pipeline_final as mg  # noqa: E402

DRIVER = "driver_decomp"


# ----------------------------------------------------------------------------- patching
def _read(p):
    with open(p) as f:
        return f.read()


def _write(p, s):
    with open(p, "w") as f:
        f.write(s)


def parse_process(cc, hh):
    """Sizes and colour data of one CPPProcess (one subprocess, nprocesses = 1)."""
    nproc = int(re.search(r"static const int nprocesses\s*=\s*(\d+)", hh).group(1))
    if nproc != 1:
        raise NotImplementedError(f"nprocesses = {nproc}: decomposition supports one process per subprocess dir")
    mats = re.findall(r"double CPPProcess::(matrix_1_\w+)\(\)", cc)
    if len(mats) != 1 or re.search(r"double CPPProcess::matrix_2_", cc):
        raise NotImplementedError(f"expected one matrix_1_* function, found {mats}")
    n_amp = int(re.search(r"static const int namplitudes\s*=\s*(\d+)", hh).group(1))
    body = cc[cc.index(f"double CPPProcess::{mats[0]}()"):]
    n_col = int(re.search(r"const int ncolor\s*=\s*(\d+)", body).group(1))
    denominators = [int(x) for x in re.search(r"const int denominators\[nprocesses\]\s*=\s*\{([^}]*)\}", cc)
                    .group(1).split(",")]
    # colour matrix: MG5 >= 3.5 writes the upper triangle packed with the off-diagonals doubled and one denom;
    # older versions the full matrix cf[ncolor][ncolor] with a denom per row
    m_packed = re.search(r"static const (?:int|double) cf\[ncolor \* \(ncolor \+ 1\)/2\]\s*=\s*\{([^}]*)\}", body)
    if m_packed:
        vals = [float(x) for x in m_packed.group(1).split(",")]
        denom = float(re.search(r"static const (?:int|double) denom\s*=\s*([\d.]+)", body).group(1))
        G = np.zeros((n_col, n_col))
        k = 0
        for i in range(n_col):
            for j in range(i, n_col):
                G[i, j] = G[j, i] = vals[k] / denom * (1.0 if i == j else 0.5)
                k += 1
    else:
        m_full = re.search(r"static const (?:int|double) cf\[ncolor\]\[ncolor\]\s*=\s*\{(.*?)\};", body, re.S)
        dens = re.search(r"static const (?:int|double) denom\[ncolor\]\s*=\s*\{([^}]*)\}", body)
        if not (m_full and dens):
            raise NotImplementedError("colour matrix format not recognised")
        vals = [float(x) for x in re.sub(r"[{}\s]", "", m_full.group(1)).split(",") if x]
        G = np.array(vals).reshape(n_col, n_col) / np.array([float(x) for x in dens.group(1).split(",")])[:, None]
        assert np.allclose(G, G.T), "unpacked colour matrix is not symmetric"
    G = G / float(denominators[0])
    return dict(matrix=mats[0], n_amp=n_amp, n_col=n_col, G=G, denominator=denominators[0])


def patch_process(sub_dir):
    """Patch CPPProcess.{h,cc} in place (on a copy): export J, accumulate X^c and X^d per helicity."""
    hh_p, cc_p = os.path.join(sub_dir, "CPPProcess.h"), os.path.join(sub_dir, "CPPProcess.cc")
    hh, cc = _read(hh_p), _read(cc_p)
    if "dc_accumulate" in cc:
        return parse_process(cc, hh)
    info = parse_process(cc, hh)
    nc, na, mat = info["n_col"], info["n_amp"], info["matrix"]
    decl = (f"  public:\n    std::complex<double> dc_jamp[{nc}];\n"
            f"    std::complex<double> dc_Xc[{nc * nc}];\n    std::complex<double> dc_Xd[{na * na}];\n"
            f"    void dc_reset();\n    void dc_accumulate();\n    void dc_coeffs(std::complex<double> * out);\n"
            f"    static const int dc_ncol = {nc};\n    static const int dc_namp = {na};\n  private:\n")
    i = hh.index("  private:")
    hh = hh[:i] + decl + hh[i + len("  private:\n"):]
    # J out of matrix_1_*: the jamp block ends where the colour sum starts
    body_i = cc.index(f"double CPPProcess::{mat}()")
    m0 = re.search(r"\n(\s*)jamp\[0\]\s*=", cc[body_i:])
    m1 = re.search(r"\n\s*double matrix\s*=\s*0", cc[body_i:])
    if not (m0 and m1):
        raise NotImplementedError("jamp block not found in " + mat)
    jamp_block = cc[body_i + m0.start():body_i + m1.start()]
    copy = f"\n  for (int dci = 0; dci < ncolor; dci++ ) dc_jamp[dci] = jamp[dci]; "
    cc = cc[:body_i + m1.start()] + copy + cc[body_i + m1.start():]
    # accumulate after every helicity evaluation in sigmaKin
    n_hook = len(re.findall(rf"t\[0\]\s*=\s*{mat}\(\);", cc))
    if n_hook == 0:
        raise NotImplementedError("no helicity loop call of " + mat)
    cc = re.sub(rf"(t\[0\]\s*=\s*{mat}\(\);)", r"\1 dc_accumulate();", cc)
    cc += f"""

// ---- decomposed targets (tools/decomp_targets.py) ----
void CPPProcess::dc_reset()
{{
  for (int k = 0; k < {nc * nc}; k++ ) dc_Xc[k] = 0.;
  for (int k = 0; k < {na * na}; k++ ) dc_Xd[k] = 0.;
}}
void CPPProcess::dc_accumulate()
{{
  for (int a = 0; a < {nc}; a++ )
    for (int b = 0; b < {nc}; b++ ) dc_Xc[a * {nc} + b] += dc_jamp[a] * conj(dc_jamp[b]);
  for (int a = 0; a < {na}; a++ )
    for (int b = 0; b < {na}; b++ ) dc_Xd[a * {na} + b] += amp[a] * conj(amp[b]);
}}
void CPPProcess::dc_coeffs(std::complex<double> * out)
{{
  // C[i][a]: MadGraph's own jamp block replayed on unit amplitudes
  const int ncolor = {nc};
  std::complex<double> jamp[ncolor];
  for (int a = 0; a < {na}; a++ )
  {{
    for (int k = 0; k < {na}; k++ ) amp[k] = 0.;
    amp[a] = 1.;
{jamp_block}
    for (int i = 0; i < ncolor; i++ ) out[i * {na} + a] = jamp[i];
  }}
}}
"""
    _write(hh_p, hh)
    _write(cc_p, cc)
    return info


def build(process, workdir):
    """Copy the production standalone of `process`, patch and compile the decomposition driver.
    Returns (sa_dir, info, driver_path)."""
    sa = mg.standalone_name(process)
    src = f"{mg.WORK_DIR}/{sa}_standalone"
    if not glob.glob(f"{src}/SubProcesses/P*/CPPProcess.cc"):
        raise FileNotFoundError(f"no standalone_cpp for {process} under {src} (datagen.ensure_backend builds it)")
    dst = os.path.join(workdir, f"{process}_decomp_sa")
    os.makedirs(workdir, exist_ok=True)
    if not os.path.isdir(dst):
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.so", "driver*", "wrapper_*", "*.o"))
    subs = mg.get_subprocess_dirs(dst)
    if len(subs) != 1:
        raise NotImplementedError(f"{process}: {len(subs)} subprocess dirs; decomposition supports one")
    sub = subs[0]
    info = patch_process(os.path.join(dst, "SubProcesses", sub))
    params = mg.parameters_class_name(dst)
    nparticles = int(mg.PROCESSES[process]["nfinal"]) + 2
    card = f"{dst}/Cards/param_card.dat"
    nc, na = info["n_col"], info["n_amp"]
    drv = f"""#include <iostream>
#include <cstdio>
#include <vector>
#include "SubProcesses/{sub}/CPPProcess.h"
// reads "<4n momenta in slot order> <aS>" per line; prints me, then C (once, first line), then per event:
// me Re/Im X^c (row-major), Re/Im X^d (row-major)
int main() {{
  CPPProcess proc;
  {{ std::streambuf* old = std::cout.rdbuf(nullptr); proc.initProc("{card}"); std::cout.rdbuf(old); }}
  std::complex<double> C[{nc * na}];
  proc.dc_coeffs(C);
  for (int k = 0; k < {nc * na}; k++ ) printf("%.17e %.17e ", real(C[k]), imag(C[k]));
  printf("\\n"); fflush(stdout);
  double p[{nparticles * 4}]; double aS;
  while (true) {{
    for (int i = 0; i < {nparticles * 4}; ++i) if (!(std::cin >> p[i])) return 0;
    if (!(std::cin >> aS)) return 0;
    {params}::getInstance()->aS = aS;
    std::vector<double*> pv;
    for (int i = 0; i < {nparticles}; ++i) pv.push_back(&p[4 * i]);
    proc.setMomenta(pv);
    proc.dc_reset();
    std::streambuf* old = std::cout.rdbuf(nullptr);
    proc.sigmaKin();
    std::cout.rdbuf(old);
    printf("%.17e", proc.getMatrixElements()[0]);
    for (int k = 0; k < {nc * nc}; k++ ) printf(" %.17e %.17e", real(proc.dc_Xc[k]), imag(proc.dc_Xc[k]));
    for (int k = 0; k < {na * na}; k++ ) printf(" %.17e %.17e", real(proc.dc_Xd[k]), imag(proc.dc_Xd[k]));
    printf("\\n"); fflush(stdout);
  }}
}}
"""
    _write(os.path.join(dst, "driver_decomp.cpp"), drv)
    srcs = [f"SubProcesses/{sub}/CPPProcess.cc", f"src/{params}.cc", "src/read_slha.cc"] + \
        sorted(os.path.relpath(p, dst) for p in glob.glob(f"{dst}/src/HelAmps_*.cc"))
    cmd = ["g++", "-O2", "-I.", "-Isrc", f"-ISubProcesses/{sub}", "driver_decomp.cpp", *srcs, "-o", DRIVER]
    r = subprocess.run(cmd, cwd=dst, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"compile failed:\n{r.stderr[-3000:]}")
    return dst, info, os.path.join(dst, DRIVER)


# ----------------------------------------------------------------------------- evaluation
def decompose(process, momenta_rows, workdir, alphas=None):
    """Decompose events (rows in the pool's order, (N, n, 4)). Returns a dict of arrays + metadata."""
    cfg = mg.PROCESSES[process]
    dst, info, drv = build(process, workdir)
    nc, na = info["n_col"], info["n_amp"]
    perm = mg.row_to_slot_perm(cfg["pdg_ids"], cfg["mg5_generate"])
    mom = np.asarray(momenta_rows, dtype=np.float64)
    if alphas is None:
        q = mom[:, 0] + mom[:, 1]
        sqrts = np.sqrt(np.clip(q[:, 0] ** 2 - (q[:, 1:] ** 2).sum(1), 0, None))
        alphas = mg.compute_alphas(sqrts, alphas_mz=float(cfg.get("alphas_mz", 0.118)))
    alphas = np.broadcast_to(np.asarray(alphas, float), (mom.shape[0],))
    lines = "\n".join(" ".join(f"{x:.17e}" for x in m[perm].reshape(-1)) + f" {a:.17e}"
                      for m, a in zip(mom, alphas)) + "\n"
    r = subprocess.run([drv], input=lines, capture_output=True, text=True, cwd=dst)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-2000:])
    out = r.stdout.strip().split("\n")
    c = np.array(out[0].split(), float).reshape(nc * na, 2)
    C = (c[:, 0] + 1j * c[:, 1]).reshape(nc, na)
    vals = np.array([l.split() for l in out[1:]], float)
    assert vals.shape == (mom.shape[0], 1 + 2 * nc * nc + 2 * na * na), vals.shape
    me = vals[:, 0]
    xc = vals[:, 1:1 + 2 * nc * nc].reshape(-1, nc * nc, 2)
    xd = vals[:, 1 + 2 * nc * nc:].reshape(-1, na * na, 2)
    Xc = (xc[..., 0] + 1j * xc[..., 1]).reshape(-1, nc, nc)
    Xd = (xd[..., 0] + 1j * xd[..., 1]).reshape(-1, na, na)
    G = info["G"]
    K = C.T @ G @ C.conj()
    sum_c = np.einsum("ij,nij->n", G, Xc).real
    sum_d = np.einsum("ab,nab->n", K, Xd).real
    return dict(me=me, Xc=Xc, Xd=Xd, G=G, C=C, K=K, sum_colour=sum_c, sum_diagram=sum_d,
                alphas=np.asarray(alphas), n_col=nc, n_amp=na)


def summarize(name, d, stored=None):
    def rel(a, b):
        return np.abs(a - b) / np.abs(b)
    msg = [f"{name}: {len(d['me'])} events, {d['n_col']} colour flows, {d['n_amp']} diagram amplitudes"]
    msg.append(f"  colour sum  vs MG5 sigmaKin: max rel {rel(d['sum_colour'], d['me']).max():.2e}")
    msg.append(f"  diagram sum vs MG5 sigmaKin: max rel {rel(d['sum_diagram'], d['me']).max():.2e}")
    if stored is not None:
        msg.append(f"  colour sum  vs stored |M|^2: max rel {rel(d['sum_colour'], stored).max():.2e}, "
                   f"median {np.median(rel(d['sum_colour'], stored)):.2e}")
        msg.append(f"  diagram sum vs stored |M|^2: max rel {rel(d['sum_diagram'], stored).max():.2e}, "
                   f"median {np.median(rel(d['sum_diagram'], stored)):.2e}")
    # how much the parts cancel: sum of |diagonal contributions| over |M|^2
    gd = np.einsum("ii,nii->n", d["G"], d["Xc"]).real / d["me"]
    kd = np.einsum("aa,naa->n", d["K"], d["Xd"]).real / d["me"]
    msg.append(f"  diagonal share of |M|^2 (median, 5%-95%): colour {np.median(gd):.3f} "
               f"[{np.quantile(gd, .05):.3f}, {np.quantile(gd, .95):.3f}], "
               f"diagram {np.median(kd):.3f} [{np.quantile(kd, .05):.3f}, {np.quantile(kd, .95):.3f}]")
    return "\n".join(msg)


def save(path, d, stored, process, pool):
    np.savez(path, stored=np.asarray(stored, float), me=d["me"],
             Xc=d["Xc"], Xd=d["Xd"], G=d["G"], C=d["C"], K=d["K"], alphas=d["alphas"],
             process=process, pool=os.path.abspath(pool) if pool else "")


def decomp_path(out_dir, pool):
    return os.path.join(out_dir, os.path.basename(pool).replace(".npy", "") + ".decomp.npz")


def run_pool(process, pool, out, workdir, n_max=None, rtol=1e-10):
    data = np.load(pool, mmap_mode="r")
    if n_max:
        data = data[:n_max]
    n = (data.shape[1] - 1) // 5
    mom = np.asarray(data[:, :4 * n]).reshape(-1, n, 4)
    stored = np.asarray(data[:, -1], float)
    d = decompose(process, mom, workdir)
    print(summarize(f"{process} [{os.path.basename(pool)}]", d, stored), flush=True)
    worst = max(np.max(np.abs(d["sum_colour"] - stored) / np.abs(stored)),
                np.max(np.abs(d["sum_diagram"] - stored) / np.abs(stored)))
    if worst > rtol:
        raise SystemExit(f"{pool}: decomposition misses the stored |M|^2 by {worst:.2e} > {rtol:g}")
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
        save(out, d, stored, process, pool)
        print(f"  -> {out}", flush=True)
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--process", required=True)
    ap.add_argument("--pool", help="one pool .npy")
    ap.add_argument("--out", help="sidecar .npz for --pool")
    ap.add_argument("--recipe", help="decompose every role's pool of this recipe (as training resolves them)")
    ap.add_argument("--out-dir", help="sidecar dir for --recipe (default $DATA_DIR/decomp)")
    ap.add_argument("--workdir", default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "decomp_sa"))
    ap.add_argument("--n-max", type=int, default=None)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--seed", type=int, default=42, help="data.seed of the runs that read the pools")
    a = ap.parse_args()
    if a.pool:
        run_pool(a.process, a.pool, a.out, a.workdir, a.n_max, a.rtol)
        return
    if not a.recipe:
        ap.error("--pool or --recipe")
    import yaml
    import datagen
    rec = yaml.safe_load(open(a.recipe))
    procs = rec["processes"] if isinstance(rec, dict) else rec
    out_dir = a.out_dir or os.path.join(os.environ.get("DATA_DIR", mg.OUTPUT_DIR), "decomp")
    # the pools as training resolves them (experiment._resolve_recipe_config): the recipe's sampling policy is
    # attached to the process before the pool path is derived; the sidecar's stored |M|^2 is checked against the
    # pool row by row when a run loads it, so a wrong pool cannot pass silently
    specs = [{"name": p["name"], "base": p.get("base", p["name"]), "physics": p.get("physics"),
              "sampling": p.get("sampling")} for p in procs]
    mg.register_recipe_processes(specs, default_sampling=rec.get("sampling") if isinstance(rec, dict) else None)
    for role, key in (("train", "n_train"), ("val", "n_val"), ("test", "n_test")):
        rs = [{"process": p["name"], "sqrts_min": float(p["sqrts"][0]), "sqrts_max": float(p["sqrts"][1]),
               "n_events": int(p[key])} for p in procs if p["name"] == a.process]
        assert rs, f"{a.process} not in {a.recipe}"
        pool = datagen.ensure_split_set(rs, role=role, seed=a.seed, require_cache=True)[a.process]
        run_pool(a.process, pool, decomp_path(out_dir, pool), a.workdir, a.n_max, a.rtol)


if __name__ == "__main__":
    main()
