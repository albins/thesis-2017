import time
import uuid
import gzip

from fabric.api import run, env, put, settings

env.hosts = ['db-manager']
clusters = ['dbnast']
env.user = "astjerna"
ssh_keys_dir = "~/astjerna-ssh-keys-{}".format(str(uuid.uuid4()))


class FabricException(Exception):
    pass


def upload_keys():
    run('mkdir -p {}'.format(ssh_keys_dir))
    run('chmod 700 {}'.format(ssh_keys_dir))
    put('id_rsa.pub', ssh_keys_dir)
    put('id_rsa', ssh_keys_dir)
    run('chmod 600 {}/id_rsa*'.format(ssh_keys_dir))


def delete_keys():
    run('rm -rf {}'.format(ssh_keys_dir))


def get_smart_data():
    upload_keys()


        #filer_command = 'system node run -node * \'priv set diag; disk shm_stats asup\''
    filer_command = 'system node run -node \\* "priv set diag; disk shm_stats asup"'
    for cluster_name in clusters:
        try:
            with settings(warn_only=True,
                          abort_exception=FabricException):
                data = run(('ssh -o PreferredAuthentications=publickey'
                            ' -i {}/id_rsa astjerna@{}-cluster-mgmt {}')
                           .format(ssh_keys_dir, cluster_name, filer_command))
                filename = "{}.{}.data.gz".format(cluster_name, time.time())
                with gzip.open(filename, mode="wb") as f:
                    f.write(str(data))
        finally:
            delete_keys()
