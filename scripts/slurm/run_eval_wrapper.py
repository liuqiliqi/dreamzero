"""Wrapper that patches the DroidJointPosClient to connect to a remote server node."""
import os
import sys

# Monkey-patch the Client class to use the server node from env vars
import sim_evals.inference.droid_jointpos as djp

SERVER_NODE = os.environ.get("SERVER_NODE", "localhost")
SERVER_PORT = int(os.environ.get("SERVER_PORT", "8000"))

_OrigClient = djp.Client
class PatchedClient(_OrigClient):
    def __init__(self, remote_host=SERVER_NODE, remote_port=SERVER_PORT, **kwargs):
        super().__init__(remote_host=remote_host, remote_port=remote_port, **kwargs)
djp.Client = PatchedClient

# Now run the eval script
exec(open("run_eval.py").read())
