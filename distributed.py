#!/usr/bin/env python3
import argparse
import os
import re
import yaml

from itertools import chain

import tensorflow as tf

import tfinput

from model import model_fn

def cli_interface():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxiter', '-i', type=int, required=True)
    parser.add_argument('--name', '-n', default='ensemble')
    parser.add_argument('--config', '-c', default='params.yml')

    args = parser.parse_args()

    return args


def main():

    # hard params to argparse later:
    args = cli_interface()
    name = args.name
    config = args.config
    maxiter = args.maxiter
    with open(config, 'r') as fd:
        cifar_params = yaml.load(fd)
    basedir = os.path.expanduser('~/distributed_cnn')
    os.makedirs(basedir, exist_ok=True)

    # network configuration
    cluster, server = configure_cluster(job_name=name)
    ntasks = cluster.num_tasks(name)

        

    for i in range(ntasks):
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:%s/task:%s' % (name, i), cluster=cluster)):
            with tf.variable_scope('task%s' % i):
                cifar_params['task'] = '_%s_%s' % (name, i)
                checkpoint = os.path.join(basedir, name + '_%s' % i)
                model = tf.estimator.Estimator(
                    model_fn=model_fn, params=cifar_params, model_dir=checkpoint
                )
                model.train(input_fn=tfinput.train_input_fn, max_steps=maxiter)


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

    cluster = tf.train.ClusterSpec({'ensemble': hosts})

    # individuals
    task = int(os.environ['SLURM_PROCID'])
    server = tf.train.Server(cluster, job_name=kwargs.get('job_name', 'job'), task_index=task)

    return cluster, server


if __name__ == '__main__':
    main()
