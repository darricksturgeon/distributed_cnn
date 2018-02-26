#!/usr/bin/env python3

from tensorflow.python.client import device_lib


def get_available_devices():

    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos]


if __name__ == '__main__':
    print(get_available_devices())
