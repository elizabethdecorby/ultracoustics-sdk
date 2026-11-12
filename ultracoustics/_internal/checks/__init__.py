"""The procedures behind the self-test and the signal level: one implementation each,
which the lab's service tools run too.

They drive the box through :class:`ultracoustics._internal.device.Box`, a Controller's
private box, never through the public Controller.
"""
