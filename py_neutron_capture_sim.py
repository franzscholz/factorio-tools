# SPDX-License-Identifier: MIT
#
# Run a simulation for the Factorio/Pyanodons Neutrion Capture process.

import argparse
import numpy as np
import simpy
from simpy.core import EmptySchedule
import logging


class Storage:
    """Simulation of the item storage containers.

    Create a synchronizer `simpy.Store` for each material.
    """

    def __init__(self, env: simpy.Environment, materials):
        """Constructor.

        Keyword arguments:
        env -- the simulation environment
        materials -- the list of materials
        """
        self.logger = logging.getLogger()
        self.env = env
        self.items = [simpy.Store(env) for m in materials]
        self.materials = materials

    def __repr__(self):
        return (
            f"(Storage(env={self.env}, items={self.items}, materials={self.materials})"  # noqa: E501
        )

    def __str__(self) -> str:
        """Print the contents of the storage."""
        # Print the storage contents
        return str(
            [(mat, len(cont.items)) for mat, cont in zip(self.materials, self.items)]  # noqa: E501
        )

    def count_items(self):
        """Get the number of items in the storages.

        Returns an array of the number of items per storage.
        """
        return [len(x.items) for x in self.items]

    def consume(self, pr):
        """Take items from the storage.

        Takes items from the storage when pr[i] is negative.

        Keyword arguments:
        pr -- Process definition (Vector of int)"""
        self.logger.debug(f"consume({pr})")
        result = []
        assert len(self.items) == len(pr)
        for i in range(len(self.items)):
            if pr[i] < 0:
                for j in range(pr[i] * -1):
                    self.logger.debug(f"get {self.materials[i]}")
                    item = self.items[i].get()
                    self.logger.debug(f"take got {item}")
                    result.append(item)
        return result

    def produce(self, pr):
        """Add items to the storage.

        Adds items to the storage when pr[i] is positive.

        Keyword arguments:
        pr -- Process definition.
        """
        self.logger.debug(f"produce({pr})")
        for i in range(len(self.items)):
            if pr[i] > 0:
                for j in range(pr[i]):
                    self.logger.debug(f"put {self.materials[i]}")
                    self.items[i].put(f"{self.materials[i]}")


class Assembler:
    """Simulation of a neutron absorber assembler."""

    def __init__(
        self,
        env: simpy.Environment,
        storage: Storage,
        process,
        duration: int,
        name: str,
    ):
        """Constructor.

        Keyword arguments:
        env -- simulation environment
        storage -- item storage
        process -- process definition
        duration -- process duration
        name - name of the process/assembler
        """
        self.logger = logging.getLogger(name)
        self.env = env
        self.storage = storage
        self.process = process
        self.duration = duration
        self.name = name
        self.p = env.process(self.neutron_capture())
        self.items = []
        self.products_made = 0
        assert np.sum(np.array(self.process)) == 0, (
            f"Process must not change item count: {self.name} {self.process}"
        )

    def __repr__(self):
        return f"Assembler({self.name}, products_made = {self.products_made}, items={self.items})"  # noqa: E501

    def neutron_capture(self):
        """Simulation process step.

        Take input items from storage, wait until available.
        Wait for the process duration.
        Add the products into the storage.
        """
        try:
            while True:
                # Take items from the store
                self.logger.debug(f"consume {self.process}")
                items_from_store = self.storage.consume(self.process)
                self.logger.debug(f"got {items_from_store}")
                # Collect all the items
                for cur in items_from_store:
                    item = yield cur
                    self.logger.debug(f"got {item}")
                    self.items.append(item)
                # Simulate procesing
                self.logger.debug(f"process for {self.duration} seconds")  # noqa: E501
                yield self.env.timeout(self.duration)
                # Put items into storage
                self.logger.debug(f"produce {self.process}")
                self.storage.produce(self.process)
                self.items = []
                # Increment counter
                self.products_made += 1
        except simpy.Interrupt as i:
            self.logger.debug(f"Interrupted: {i.cause}")


def main():
    # Process definitions.
    # Index: 238/239/240/241/242.
    # Positive: output, Negative: input
    # Output of the PU-Oxide procss.
    puox = np.array([2, 53, 25, 15, 50])
    # Neutron capture processes
    nc_240 = [0, -1, 1, -1, 1]
    nc_238 = [1, -1, -1, 0, 1]
    nc_241 = [0, 1, 0, 1, -2]
    nc_239 = [0, 1, 1, -1, -1]
    materials = ["PU-238", "PU-239", "PU-240", "PU-241", "PU-242"]
    process_names = ["PU-240", "PU-238", "PU-241", "PU-239"]
    # Production matrix
    nc = np.array([nc_240, nc_238, nc_241, nc_239])
    # Production matrix transposed
    # nc_t = nc.transpose()
    # Duration of each process in seconds
    duration = np.array([442, 202, 281, 552])

    # Result of the Jupyter notebook
    machines_manual = [0, 4, 3, 6]
    # machines_manual = [1, 0, 0, 0]

    # Simulation
    env = simpy.Environment()

    # Initialize the item storage
    storage = Storage(env, materials)

    def check_items(storage: Storage, assemblers):
        """Check if the number of items satisfies completeness.

        There must be enough final products and at least two of
        the unneeded product storages must be empty.

        Returns true if done.
        """
        c = storage.count_items()
        sum_zero = (
            (1 if c[2] == 0 else 0) + (1 if c[3] == 0 else 0) + (1 if c[4] == 0 else 0)  # noqa: E501
        )
        sum_assembler_items = 0
        for cur in assemblers:
            sum_assembler_items += len(cur.items)
        logging.debug(
            f"check_items: c={c} sum_zero={sum_zero} sum_assembler_items={sum_assembler_items}"  # noqa: E501
        )
        return c[0] > 0 and c[1] > 0 and sum_zero >= 2 and sum_assembler_items == 0  # noqa: E501

    # Add the initial batch to the storage
    def produce(env: simpy.Environment, storage: Storage, process):
        """The PU-Oxide production process.

        This process produces the inputs to the storage.
        """
        logging.debug(f"produce({process})")
        # Add products to the storage.
        storage.produce(process)
        logging.debug(f"Storage: {storage}")
        yield env.timeout(1)

    # Start producer
    prod = env.process(produce(env, storage, puox))  # noqa: F841

    # Initialize the machines
    assemblers = []
    for cur in zip(machines_manual, nc, duration, process_names):
        for j in range(int(cur[0])):
            m = Assembler(env, storage, cur[1], cur[2], f"{cur[3]} #{j}")
            assemblers.append(m)

    # Run the simulation until done
    try:
        while not check_items(storage, assemblers):
            env.step()
        logging.debug("check_items: Simulation complete.")
    except EmptySchedule:
        logging.debug("EmptySchedule: Simulation complete.")

    # Print the result of the simulation
    print(f"Simulation complete after {env.now} seconds.")
    logging.debug(f"Storage: {storage}")
    logging.debug(f"Machines: {assemblers}")
    final_items = np.sum(np.array(storage.count_items()))
    final_items += np.sum(np.array([len(cur.items) for cur in assemblers]))
    if np.sum(puox) != final_items:
        logging.error(
            f"Input item count {np.sum(puox)} != output item count {final_items}"  # noqa: E501
        )


if __name__ == "__main__":
    # Initialize logging
    logging.basicConfig()
    # Setup argument parser and parse command line arguments
    parser = argparse.ArgumentParser(
        prog="py_neutron_capture_sim",
        description="Simulate Factorio/Pyanodons neutron capture.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")  # noqa: E501
    parser.add_argument("-d", "--debug", action="store_true", help="Debug output")  # noqa: E501
    args = parser.parse_args()

    # Configure logging.
    # If running under a debugger set the loglevel to DEBUG.
    def check_debug() -> bool:
        import sys

        has_trace = hasattr(sys, "gettrace") and sys.gettrace() is not None
        has_breakpoint = sys.breakpointhook.__module__ != "sys"
        isdebug = has_trace or has_breakpoint
        return isdebug

    if check_debug():
        logging.getLogger().setLevel(logging.DEBUG)
    # Set the loglevel according to the command line options
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    # Run simulation
    main()
