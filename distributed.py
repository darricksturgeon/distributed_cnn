#!/usr/bin/env python3
import os
import re

from itertools import chain

import tensorflow as tf

import tfinput

from model import model_fn


def main():

    # hard params to argparse later:
    name = 'ensemble_cnn'
    cifar_params = {
        'img_dim': [32, 32, 3],
        'y_size': 10,
        'learning_rate': .001
    }
    basedir = os.path.expanduser('~/distributed_cnn')
    os.makedirs(basedir, exist_ok=True)

    # network configuration
    cluster, server = configure_cluster(job_name=name)
    ntasks = cluster.num_tasks(name)
    models = {}

    for i in range(ntasks):
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:workers/task:' + i, cluster=cluster)):
            checkpoint = os.path.join(basedir, name + '_%s' % i)
            models['task%s' % i] = tf.estimator.Estimator(
                model_fn=model_fn, params=cifar_params, model_dir=checkpoint
            )
            models['task%s' % i].train(input_fn=tfinput.train_input_fn, max_steps=1000)




def configure_cluster(**kwargs):
    # unpack the concatenated list of nodes returned
    def rangeString(commaString):
        # https://stackoverflow.com/questions/6405208/how-to-convert-numeric-string-ranges-to-a-list-in-python
        def hyphenRange(hyphenString):
            x = [int(x) for x in hyphenString.split('-')]
            return range(x[0], x[-1] + 1)

        return chain(*[hyphenRange(r) for r in commaString.split(',')])

    # cluster properties
    nodelist = os.environ['SLURM_JOB_NODELIST']
    hostexpr = re.compile(r"(.*)\[(.*)\]")
    nodegroups = hostexpr.match(nodelist)
    basename, noderange = nodegroups.groups()
    nodenums = rangeString(noderange.strip('[]'))

    hosts = [basename + str(n) + ':2222' for n in nodenums]

    cluster = tf.train.ClusterSpec({'workers': hosts})

    # individuals
    task = int(os.environ['SLURM_PROCID'])
    server = tf.train.Server(cluster, job_name=kwargs.get('job_name', 'job'), task_index=task)

    return cluster, server


if __name__ == '__main__':
    main()
