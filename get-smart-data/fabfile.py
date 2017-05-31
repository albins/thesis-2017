import time
import gzip
import os.path

from fabric.api import run, env, parallel

clusters = ['dbnast', 'dbnasr']
env.hosts = ["{}-cluster-mgmt".format(c) for c in clusters]
env.user = "astjerna"
env.key_filename = "id_rsa"
data_store_dir = ""


def on_cluster(node_exp, command):
    return 'system node run -node {} "{}"'.format(node_exp, command)


@parallel
def get_smart_data():
    filer_command = on_cluster(node_exp="*",
                               command="priv set diag; disk shm_stats asup")
    data = run(filer_command, shell=False)
    filename = os.path.join(data_store_dir,
                            "{}.{}.data.gz".format(env.host, time.time()))
    with gzip.open(filename, mode="wb") as f:
        f.write(str(data))
