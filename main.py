import argparse

import tensorflow as tf


def cli():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return interface(**vars(args))


def interface(**kwargs):
    players = kwargs.get('players', ['localhost:2222', 'localhost:2223', 'localhost:2224'])
    ps = kwargs.get('ps', ['localhost:2225'])
    cluster = tf.train.ClusterSpec(
        {'players': players,
         'ps': ps
         }
    )

    ps = []

    workers = []
    for i, worker in enumerate(players):
        workers.append()

def run_layer(worker):
    pass


if __name__ == '__main__':
    cli()
