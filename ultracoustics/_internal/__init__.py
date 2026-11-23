"""
PRIVATE. Not part of the SDK's API, and not supported. Do not import from here.

This package is what the public calls run on: the USB stream, the box's commands,
and the procedures behind the self-test, the signal level, the lock and firmware
updates. It ships because they need it, not for customers to call.

Code here can drive the lasers, and write firmware and the settings stored on the
box, without the checks the public calls make first. A mistake can leave the box
needing service, with no recovery over USB. Names and behaviour change without
notice, and Ultracoustics does not support a box that has been driven through it.

Use :class:`ultracoustics.Controller` and the other names on :mod:`ultracoustics`.
If something you need is missing there, ask Ultracoustics.
"""
