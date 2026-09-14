# USB OUT recovery and migration contract

An OUT write result describes transport only. `delivered` means the host USB
stack reported the full payload transferred; it does not mean firmware parsed,
accepted, or applied the request. `uncertain` means some or all of the nonempty
payload may have reached the device. Pre-write validation/lifecycle failures can
be definite `failed` results; USB write exceptions are conservatively uncertain.

On PIPE, the SDK clears the endpoint halt once where possible and never
automatically replays the payload. A generic PyUSB/python-libusb1 exception has
no independently verified transferred-byte count. A successful halt clear says
only that later explicit requests may be attempted. It cannot identify which
earlier request firmware rejected. Persistent PIPE is therefore bounded by the
same rule: one write, at most one halt clear, no retry.

A successful zero-length USB transfer is a real zero-length packet. It is not
evidence that a failed nonempty transfer moved zero bytes.

Direct and streaming callers receive the same semantics. Direct `send()`
returns a transport result on full delivery. Confirmed streaming sends return a
result with `transport_status`; uncertain sends raise `TransportUncertainError`
and other transport failures raise `RuntimeError`. `CommandRejectedError`
remains an import/catch alias, but its old application-level interpretation is
deprecated. Queue acknowledgement timestamps bound host-side completion only.

## Caller migration

- Catch `TransportUncertainError`, record the event, and stop the current
  stateful workflow. Do not infer rejection or success.
- Never automatically replay firmware BEGIN/page/COMMIT, power, trigger, DAC,
  or parameter commands after PIPE, timeout, short transfer, or disconnect.
- For firmware update uncertainty, reconnect and verify documented device/image
  state before a human- or protocol-directed resume. Page retry needs a future
  application acknowledgement/idempotency contract.
- Read-only/idempotent retry is not enabled generically. A future method may add
  one bounded retry only when the operation is explicitly classified and its
  response/request identity makes the retry safe.
- Future manual control requires request IDs and received/applied/rejected
  application acknowledgements. Transport status must not be relabelled as an
  application acknowledgement.
