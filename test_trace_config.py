import struct
from types import SimpleNamespace
import pytest
from ultracoustics.system_control import SystemControlMixin
from ultracoustics._internal.control import MANUAL_GET, MANUAL_SET
from ultracoustics._internal.optical_diagnostics import parse_page, crc16_ccitt, OpticalDiagnosticError
from test_retained_trace import raw, RetainedTraceTests

class Fake(SystemControlMixin):
    def __init__(self):
        self.word=0x400082
        self.calls=[]
    def _normalized_pi_command_638(self, opcode, channel, value=0, timeout_s=1):
        assert channel==13
        self.calls.append((opcode,value))
        if opcode==MANUAL_SET:self.word=value
        return SimpleNamespace(status=0, applied_value=self.word)

def extended(index=0, amp=8, hold=40, decimation=10, injection=0):
    b=bytearray(raw(index))
    struct.pack_into('<H',b,16,5|((decimation-1)<<4)|(0x400 if decimation>1 else 0))
    for offset in (28,40):struct.pack_into('<h',b,offset,injection)
    b[48:50]=bytes([amp,hold])
    struct.pack_into('<H',b,50,crc16_ccitt(b[:50]))
    return bytes(b)

def test_config_roundtrip_and_bounds():
    f=Fake()
    r=f.configure_control_trace_638(64,255,64)
    assert (r['amplitude_dac'],r['hold_updates'],r['decimation'])==(64,255,64)
    assert [x[0] for x in f.calls]==[MANUAL_GET,MANUAL_SET,MANUAL_GET]
    for args in [(0,1,1),(65,1,1),(2,0,1),(2,256,1),(2,1,65),(True,1,1),(2,1.1,1)]:
        before=len(f.calls)
        with pytest.raises(ValueError):f.configure_control_trace_638(*args)
        assert len(f.calls)==before
    for word in [0,0x600082,0xc00082,0x400000,0x4000c1]:
        with pytest.raises(RuntimeError):f._decode_trace_config(word)

def test_extended_pages_and_legacy():
    p=parse_page(extended())
    assert (p.amplitude_dac,p.hold_updates,p.decimation,p.recorder_filtered)==(8,40,10,True)
    old=parse_page(raw(0))
    assert (old.amplitude_dac,old.hold_updates,old.decimation,old.recorder_filtered)==(2,None,1,False)
    for args in [dict(amp=65),dict(hold=0),dict(injection=9),dict(decimation=1,injection=0)]:
        with pytest.raises(OpticalDiagnosticError):parse_page(extended(**args))
    for mask in [0x800,0x400]:
        b=bytearray(extended());flags=struct.unpack_from('<H',b,16)[0]^mask
        struct.pack_into('<H',b,16,flags);struct.pack_into('<H',b,50,crc16_ccitt(b[:50]))
        with pytest.raises(OpticalDiagnosticError):parse_page(b)

class ExtendedRetention(RetainedTraceTests):
    def test_extended_metadata_and_conflict(self):
        self.put(extended(0));self.put(extended(2))
        from ultracoustics._internal.telemetry import read_retained_trace
        r=read_retained_trace(self.shm)
        assert r['complete'] and r['amplitude_dac']==8 and r['hold_updates']==40
        assert r['recorder_filtered'] and r['decimation']==10
        self.put(extended(0,amp=9))
        assert read_retained_trace(self.shm)['conflict']
