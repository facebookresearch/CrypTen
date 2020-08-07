import multiprocessing
import os
import uuid

import crypten.communicator as comm
import crypten.mpc.primitives.baseOT as baseOT


def init_process(env, rank):
    for env_key, env_value in env.items():
        os.environ[env_key] = env_value
    os.environ["RANK"] = str(rank)

    comm._init(use_threads=False)

    ot = baseOT.BaseOT(1 - rank)
    if rank == 0:
        msg0s = ["123", "abc"]
        msg1s = ["def", "123"]
        ot.send(msg0s, msg1s, 2)
        choices = [1, 0]
        msgcs = ot.receive(choices, 2)
        print(rank, msgcs)
    else:
        choices = [0, 1]
        msgcs = ot.receive(choices, 2)
        msg0s = ["xyz", "123"]
        msg1s = ["123", "uvw"]
        ot.send(msg0s, msg1s, 2)
        print(rank, msgcs)


if __name__ == "__main__":
    env = os.environ.copy()
    env["WORLD_SIZE"] = str(2)
    multiprocessing.set_start_method("spawn")

    # Use random file so multiple jobs can be run simultaneously
    INIT_METHOD = "file:///tmp/crypten-rendezvous-{}".format(uuid.uuid1())
    env["RENDEZVOUS"] = INIT_METHOD
    processes = []
    for rank in range(2):
        process_name = "process " + str(rank)
        process = multiprocessing.Process(
            target=init_process, name=process_name, args=(env, rank)
        )
        processes.append(process)
        process.start()

    for rank in range(2):
        processes[rank].join()
