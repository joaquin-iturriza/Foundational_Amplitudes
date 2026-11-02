"""
Exclusive lock between jobs on a shared filesystem, built on an atomic mkdir (ported from
~/work/likelihoods sweep/dirlock.py, 128f913).

fcntl.flock cannot be trusted where FA's sweeps and caches live: EOS has no advisory locks, and on CC-IN2P3's /sps
(NFS) some compute nodes answer every flock with ENOLCK ("No locks available"); a blocking flock there killed a
width run and 7 sweep trials at the memmap cache lock (catalog_coverage, 2026-11-02, E3550), and the likelihoods
sweeps lost 4 of 20 trials at their first suggest(). mkdir of an existing directory fails atomically on all of them,
so the lock is the directory <path>.lockdir; the holder writes host and pid into it and, from a heartbeat thread,
touches it every HEARTBEAT_S. A lock not touched for STALE_S (its holder was killed) is broken by renaming it away,
which only one waiter can do. The heartbeat lets a holder keep the lock for as long as it needs (the memmap cache
build takes about 20 min) while a dead holder's lock is broken within STALE_S.
"""

import os
import socket
import threading
import time
import uuid
from contextlib import contextmanager

STALE_S = 600.0
HEARTBEAT_S = 30.0
POLL_S = 2.0


def _beat(lock, stop, every):
    # only a vanished lock ends the heartbeat: a transient NFS error (ESTALE, EIO) must not, or a live holder's lock
    # goes stale and a waiter breaks it
    while not stop.wait(every):
        try:
            os.utime(lock)
        except FileNotFoundError:
            return
        except OSError:
            pass


@contextmanager
def dir_lock(path, stale_s=STALE_S, poll_s=POLL_S, heartbeat_s=HEARTBEAT_S, on_wait=None, sleep=time.sleep):
    """Hold <path>.lockdir for the body. on_wait(owner_line) is called once when the lock is found held."""
    lock = path + ".lockdir"
    told = False
    while True:
        try:
            os.mkdir(lock)
            break
        except FileExistsError:
            try:
                age = time.time() - os.stat(lock).st_mtime
            except FileNotFoundError:
                continue
            if age > stale_s:
                # break it by renaming it away; the renamed dir must still be the stale lock we judged (same owner
                # token, still old): another waiter may have broken it first and taken a fresh lock in its place,
                # which is then put back
                seen = _owner(lock)
                stale = "%s.stale.%s.%d.%s" % (lock, socket.gethostname(), os.getpid(), uuid.uuid4().hex[:8])
                try:
                    os.rename(lock, stale)
                except OSError:
                    continue
                try:
                    still = _owner(stale) == seen and time.time() - os.stat(stale).st_mtime > stale_s
                except OSError:
                    still = False
                if still:
                    _rmtree(stale)
                else:
                    try:
                        os.rename(stale, lock)
                    except OSError:
                        pass
                continue
            if on_wait is not None and not told:
                told = True
                on_wait(_owner(lock))
            sleep(poll_s)
    token = "%s %d %s %.0f" % (socket.gethostname(), os.getpid(), uuid.uuid4().hex, time.time())
    stop = threading.Event()
    beat = threading.Thread(target=_beat, args=(lock, stop, heartbeat_s), daemon=True)
    try:
        with open(os.path.join(lock, "owner"), "w") as f:
            f.write(token + "\n")
        beat.start()
        yield
    finally:
        stop.set()
        if beat.is_alive():
            beat.join()
        if _owner(lock) == token:       # never remove a lock that is no longer ours
            _rmtree(lock)


def _owner(lock):
    try:
        with open(os.path.join(lock, "owner")) as f:
            return f.read().strip()
    except OSError:
        return "?"


def _rmtree(d):
    try:
        names = os.listdir(d)
    except FileNotFoundError:
        return
    for name in names:
        try:
            os.remove(os.path.join(d, name))
        except FileNotFoundError:
            pass
    try:
        os.rmdir(d)
    except FileNotFoundError:
        pass
