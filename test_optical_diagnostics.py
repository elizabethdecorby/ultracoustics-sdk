import struct
import pytest
from ultracoustics._internal.optical_diagnostics import *

def page(kind):
    b=bytearray(52);struct.pack_into("<BBH",b,0,1,kind,52);struct.pack_into("<H",b,50,crc16_ccitt(b[:50]));return b
def test_crc_rejected():
    b=page(PAGE_LIVE);b[8]^=1
    with pytest.raises(OpticalDiagnosticError):parse_page(b)
def test_pages_latch_independently_and_epoch_resets():
    c=OpticalDiagnosticCache();c.publish(page(PAGE_LIVE),1,10,100);c.publish(page(PAGE_ACQUISITION),1,11,101)
    assert c.live.record_sequence==10 and c.acquisition.record_sequence==11 and c.abba is None
    c.publish(page(PAGE_ABBA),2,12,102)
    assert c.live is None and c.acquisition is None and c.abba.record_sequence==12
def test_wrap_values_decode_unsigned():
    b=page(PAGE_LIVE);struct.pack_into("<II",b,4,0xffffffff,0xfffffffe);struct.pack_into("<H",b,50,crc16_ccitt(b[:50]))
    v=parse_page(b);assert v.sequence==0xffffffff and v.tick_ms==0xfffffffe
def test_live_timing_extension_is_optional_and_decodes_when_flagged():
    old=parse_page(page(PAGE_LIVE))
    assert old.bad_feedback_frames is None
    assert old.dac_busy_completions is None
    assert old.control_exec_max_us is None
    assert old.frame_interval_max_us is None

    b=page(PAGE_LIVE)
    b[14]=0x08
    struct.pack_into("<HHHH",b,42,0xffff,2,37,401)
    struct.pack_into("<H",b,50,crc16_ccitt(b[:50]))
    new=parse_page(b)
    assert new.bad_feedback_frames==0xffff
    assert new.dac_busy_completions==2
    assert new.control_exec_max_us==37
    assert new.frame_interval_max_us==401
def test_trace_page_decodes_records_and_checks_bounds():
    b=page(PAGE_TRACE)
    struct.pack_into("<IHHIHBB",b,4,7,4094,4096,144_000_000,5,2,12)
    struct.pack_into("<IHHhH",b,20,100,6000,20000,-2,1)
    struct.pack_into("<IHHhH",b,32,200,6001,20002,2,1)
    struct.pack_into("<I",b,44,1234)
    struct.pack_into("<H",b,50,crc16_ccitt(b[:50]))
    trace=parse_page(b)
    assert trace.capture_id==7 and trace.start_index==4094
    assert trace.samples[0].injection_dac==-2 and trace.samples[1].actual_dac==20002
    bad=bytearray(b);struct.pack_into("<H",bad,10,4095);struct.pack_into("<H",bad,50,crc16_ccitt(bad[:50]))
    with pytest.raises(OpticalDiagnosticError):parse_page(bad)
    cache=OpticalDiagnosticCache()
    cache.publish(page(PAGE_ABBA),1,1,100)
    cache.publish(b,1,2,101)
    assert cache.abba is not None and cache.trace.page.capture_id==7
    cache.publish(page(PAGE_LIVE),2,3,102)
    assert cache.trace is None and cache.abba is None

def test_scan_trace_flag_and_limit():
    b=page(PAGE_TRACE)
    struct.pack_into("<IHHIHBB",b,4,8,0,1000,144_000_000,9,2,12)
    struct.pack_into("<IHHhH",b,20,100,6000,0,0,1)
    struct.pack_into("<IHHhH",b,32,200,6001,50,0,1)
    struct.pack_into("<H",b,50,crc16_ccitt(b[:50]))
    assert parse_page(b).flags==9
    struct.pack_into("<H",b,10,1003)
    struct.pack_into("<H",b,50,crc16_ccitt(b[:50]))
    with pytest.raises(OpticalDiagnosticError):parse_page(b)
