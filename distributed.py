#!/usr/bin/env python3
import argparse
import json
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
    parser.add_argument('--name', '-n', default='worker')
    parser.add_argument('--params', '-c', default='params.yml')

    args = parser.parse_args()

    return args


def main():

    args = cli_interface()
    name = args.name
    params = args.params
    maxiter = args.maxiter
    with open(params, 'r') as fd:
        cifar_params = yaml.load(fd)
    basedir = os.path.expanduser('~/distributed_cnn')
    os.makedirs(basedir, exist_ok=True)

    # network configuration
    cluster, server, task = configure_cluster(job_name=name)
    ntasks = cluster.num_tasks(name)
    os.environ['TF_CONFIG'] = json.dumps(
        {'cluster': cluster.as_dict(),
         'task': {'type': name, 'index': task}}
    )

    for i in range(ntasks):
        with tf.device(tf.train.replica_device_setter(
                worker_device='/job:%s/task:%s' % (name, i), cluster=cluster)):
            with tf.variable_scope('task%s' % i):
                conf = tf.estimator.RunConfig(
                    model_dir='ensemble%s_epoch%s' % (i, maxiter),
                    tf_random_seed=i
                )
                cifar_params['task'] = '_%s_%s' % (name, i)
                checkpoint = os.path.join(basedir, name + '_%s' % i)
                model = tf.estimator.Estimator(
                    model_fn=model_fn, params=cifar_params, model_dir=checkpoint, config=conf
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

    cluster = tf.train.ClusterSpec({kwargs.get('job_name', 'worker'): hosts})

    # individuals
    task = int(os.environ['SLURM_PROCID'])
    server = tf.train.Server(cluster, job_name=kwargs.get('job_name', 'worker'), task_index=task)

    return cluster, server, task


if __name__ == '__main__':
    main()
