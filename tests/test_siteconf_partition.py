"""A sweep's cluster block may narrow the site's partition list to some of the site's own partitions; any other
partition is dropped and the site's list is kept."""
import siteconf


def _part(cluster):
    siteconf.CLUSTER.update({"partition": "gpu_v100,gpu_h100", "scheduler": "slurm"})
    cl = siteconf.resolve({"cluster": dict(cluster)})["cluster"]
    return [l for l in siteconf.slurm_header(cl, "t", "o", "e").split("\n") if "--partition" in l]


def test_narrowed():
    assert _part({"partition": "gpu_h100"}) == ["#SBATCH --partition=gpu_h100"]


def test_default_and_foreign():
    assert _part({}) == ["#SBATCH --partition=gpu_v100,gpu_h100"]
    assert _part({"partition": "gpu_p5"}) == ["#SBATCH --partition=gpu_v100,gpu_h100"]


if __name__ == "__main__":
    test_narrowed(); test_default_and_foreign(); print("ok")
