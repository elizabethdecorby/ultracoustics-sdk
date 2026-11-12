"""Firmware versions and identity records of all three boards.

The reads are maintenance queries, so they need the USB device to themselves:
close any Controller first. The timing was tuned on hardware, as the lab's
diagnostics scan uses it: the master needs a gap after every slave query, and
the 638 ignores the first query after it powers up.
"""
import time

#: Before every identity read: a slave query keeps the master busy for 2 s.
IDENTITY_SETTLE_S = 3.0
IDENTITY_SLAVE_ATTEMPTS = 2

BOARD_LABELS = {'main': 'Main board', '1550': '1550 nm laser board',
                '638': '638 nm laser board'}
#: Identity fields each board stores.
IDENTITY_FIELDS = {
    'main': (('serial', 'Serial'), ('pcb_rev', 'CPU PCB'),
             ('pd_pcb_rev', 'PD PCB'), ('circ_sn', 'Circulator S/N')),
    '1550': (('serial', 'Serial'), ('pcb_rev', 'PCB'),
             ('driver_pot_ohm', 'Driver pot'), ('lpf_pot_ohm', 'LPF pot'),
             ('pd_pot_ohm', 'PD pot')),
}
IDENTITY_FIELDS['638'] = IDENTITY_FIELDS['1550']
IDENTITY_STATUS = {
    'EMPTY': 'no record stored',
    'NO_RESPONSE': 'the board did not answer (no board, or a bootloader '
                   'without identity support)',
    'BAD_STATE': 'the main board was not in IDLE; STOP first',
}


def read_device_info(progress=lambda message: None, sleep=time.sleep,
                     clock=time.monotonic, connection=None):
    """Read every board's firmware version and identity record.

    Returns ``{'usb_serial', 'firmware', 'identity', 'warnings'}``:
    ``firmware`` maps ``'main'`` to the master's firmware info and
    ``'1550'`` / ``'638'`` to ``{'boot', 'app'}``; ``identity`` maps each
    board to its ``IdentityResult``. A query that failed leaves its board
    out and says why in ``warnings``. Raises only when the master cannot be
    opened. Needs the main board in IDLE.
    """
    from ..comms import USBBulkConnection
    from ..maintenance import DiagnosticsManager

    conn = connection or USBBulkConnection()
    info = {'usb_serial': None, 'firmware': {}, 'identity': {}, 'warnings': []}
    try:
        try:
            info['usb_serial'] = conn.dev.serial_number
        except Exception as exc:
            info['warnings'].append(f'USB serial unreadable: {exc}')
        _read(DiagnosticsManager(conn), info, progress, sleep, clock)
    finally:
        conn.close()
    return info


def _read(diag, info, progress, sleep, clock):
    from ...updates import read_firmware_versions
    from ..protocol import TARGET_1550, TARGET_638, TARGET_MAIN

    firmware, warnings = read_firmware_versions(diag, progress, sleep, clock)
    info['firmware'], info['warnings'] = firmware, warnings
    visible = (firmware.get('main') or {}).get('profile') == 'visible_pd'

    for board, target in (('main', TARGET_MAIN), ('1550', TARGET_1550),
                          ('638', TARGET_638)):
        if visible and board == '1550':
            continue
        attempts = 1 if board == 'main' else IDENTITY_SLAVE_ATTEMPTS
        for attempt in range(1, attempts + 1):
            sleep(IDENTITY_SETTLE_S)
            progress(f'Reading the {BOARD_LABELS[board].lower()} identity…')
            try:
                result = diag.read_identity(target)
            except Exception as exc:
                if board == 'main':
                    # The master answers every identity command it accepts,
                    # so the slaves would not answer either.
                    warnings.append(f'Main board identity: {exc}')
                    return
                if attempt == attempts:
                    warnings.append(f'{BOARD_LABELS[board]} identity: {exc}')
                continue
            info['identity'][board] = result
            if result.status_name != 'NO_RESPONSE':
                break


def firmware_text(fw):
    """One line for a board's firmware info."""
    if not fw:
        return 'not read'
    parts = [(fw.get('version') or '').strip() or 'unnamed']
    if fw.get('hardware_rev'):
        parts.append(f"hw rev {fw['hardware_rev']}")
    if fw.get('build_date') is not None:
        parts.append(f"built {fw['build_date']:%Y-%m-%d %H:%M}")
    if fw.get('profile') == 'visible_pd':
        parts.append('visible-PD build')
    return ', '.join(parts)


def field_text(name, value):
    """An identity field's value as shown."""
    if value is None:
        return '—'
    if name.endswith('_ohm'):
        return f'{value:g} Ω'
    return str(value)


def format_device_info(info, statuses=()):
    """``read_device_info()``'s result as text: one block per board, then any warnings.
    ``statuses``: each board against the published list (``ultracoustics.updates``),
    when known."""
    published = {status.board: status for status in statuses}
    lines = [f"USB serial      {info.get('usb_serial') or 'unknown'}"]
    for board, label in BOARD_LABELS.items():
        lines += ['', label]
        fw = info['firmware'].get(board)
        if board == 'main':
            lines.append(f'  Firmware      {firmware_text(fw)}')
        else:
            lines.append(f"  Bootloader    {firmware_text((fw or {}).get('boot'))}")
            lines.append(f"  Application   {firmware_text((fw or {}).get('app'))}")
        if board in published:
            lines.append(f'  Published     {published[board].describe()}')
        result = info['identity'].get(board)
        if result is None:
            lines.append('  Identity      not read')
            continue
        if result.record is None:
            status = result.status_name
            lines.append(f'  Identity      {IDENTITY_STATUS.get(status, status)}')
            continue
        identity = result.record.identity
        written = (time.strftime('%Y-%m-%d %H:%M', time.localtime(identity.programmed_at))
                   if identity.programmed_at else 'time not recorded')
        lines.append(f'  Identity      record {result.record.seq}, written {written}'
                     + (f', report {identity.report_id}' if identity.report_id else ''))
        for name, field_label in IDENTITY_FIELDS[board]:
            lines.append(f'    {field_label:<16}{field_text(name, getattr(identity, name))}')
    if info['warnings']:
        lines += ['', 'Warnings'] + [f'  {w}' for w in info['warnings']]
    return '\n'.join(lines)
