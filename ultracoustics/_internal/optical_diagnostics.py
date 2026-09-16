"""Pure decoder and bounded cache for 638 optical diagnostic pages."""
from __future__ import annotations
from dataclasses import dataclass, replace
import struct
import time
from typing import Optional, Union

PAGE_BYTES=52
VERSION=1
PAGE_LIVE=2
PAGE_ACQUISITION=3
PAGE_ABBA=4

class OpticalDiagnosticError(ValueError): pass

def crc16_ccitt(data: bytes) -> int:
    crc=0xffff
    for value in data:
        crc ^= value << 8
        for _ in range(8): crc=((crc<<1)^0x1021)&0xffff if crc&0x8000 else (crc<<1)&0xffff
    return crc

@dataclass(frozen=True)
class OpticalLive:
    sequence:int; tick_ms:int; state:int; phase:int; flags:int; gain_law:int
    dac:int; feedback:int; target:int; error:int; slope:float; kp:float; ki:float
    clip_count:int; quality_count:int; timing_min_us:int; timing_max_us:int
    bad_feedback_frames:Optional[int]=None
    dac_busy_completions:Optional[int]=None
    control_exec_max_us:Optional[int]=None
    frame_interval_max_us:Optional[int]=None
@dataclass(frozen=True)
class OpticalAcquisition:
    event_id:int; start_tick_ms:int; end_tick_ms:int; baseline:int; minimum:int; target:int
    lower_dac:int; upper_dac:int; best_feedback:int; final_dac:int
    iterations:int; hit:bool; result:int; retry_reason:int
@dataclass(frozen=True)
class OpticalABBA:
    event_id:int; start_tick_ms:int; end_tick_ms:int; readings:tuple[int,int,int,int]
    offset_dac:int; settle_ms:int; average_ms:int; slope:float; residual:float
    result:int; old_gain_law:int; new_gain_law:int
    old_kp:float; old_ki:float; new_kp:float; new_ki:float
OpticalPage=Union[OpticalLive,OpticalAcquisition,OpticalABBA]

def parse_page(raw:bytes)->OpticalPage:
    if len(raw)!=PAGE_BYTES: raise OpticalDiagnosticError("optical page must be exactly 52 bytes")
    version,page_type,length=struct.unpack_from("<BBH",raw)
    if version!=VERSION or length!=PAGE_BYTES: raise OpticalDiagnosticError("unsupported optical page header")
    if crc16_ccitt(raw[:50])!=struct.unpack_from("<H",raw,50)[0]: raise OpticalDiagnosticError("optical page CRC mismatch")
    if page_type==PAGE_LIVE:
        seq,tick=struct.unpack_from("<II",raw,4); state,phase,flags,law=struct.unpack_from("<BBBB",raw,12)
        dac,fb,target,error,slope,kp,ki=struct.unpack_from("<HHHhhhh",raw,16)
        clip,quality=struct.unpack_from("<II",raw,30); tmin,tmax=struct.unpack_from("<HH",raw,38)
        timing_diagnostics = struct.unpack_from("<HHHH", raw, 42) if flags & 0x08 else (None,) * 4
        return OpticalLive(seq,tick,state,phase,flags,law,dac,fb,target,error,slope/256,kp/4096,ki/4096,clip,quality,tmin,tmax,*timing_diagnostics)
    if page_type==PAGE_ACQUISITION:
        vals=struct.unpack_from("<IIIHHHHHHHBBBB",raw,4)
        return OpticalAcquisition(*vals[:10], vals[10], bool(vals[11]), *vals[12:])
    if page_type==PAGE_ABBA:
        event,start,end,*readings=struct.unpack_from("<IIIHHHH",raw,4)
        offset,settle,average=struct.unpack_from("<hHH",raw,24); slope,residual=struct.unpack_from("<ii",raw,30)
        result,oldlaw,newlaw=struct.unpack_from("<BBB",raw,38); gains=struct.unpack_from("<hhhh",raw,42)
        return OpticalABBA(event,start,end,tuple(readings),offset,settle,average,slope/65536,residual/65536,result,oldlaw,newlaw,*(v/4096 for v in gains))
    raise OpticalDiagnosticError(f"unknown optical page type {page_type}")

@dataclass(frozen=True)
class CachedOpticalPage:
    page:OpticalPage; stream_epoch:int; record_sequence:int; received_monotonic_ns:int
    @property
    def host_age_s(self)->float: return max(0,time.monotonic_ns()-self.received_monotonic_ns)/1e9
    @property
    def host_stale(self)->bool: return time.monotonic_ns()-self.received_monotonic_ns > 500_000_000

class OpticalDiagnosticCache:
    """Keeps the newest page of each type; an arriving page never clears peers."""
    def __init__(self): self.live=None; self.acquisition=None; self.abba=None; self._epoch=None
    def publish(self,raw:bytes,stream_epoch:int,record_sequence:int,received_monotonic_ns:Optional[int]=None):
        page=parse_page(raw); received=time.monotonic_ns() if received_monotonic_ns is None else received_monotonic_ns
        if self._epoch is not None and stream_epoch!=self._epoch: self.live=self.acquisition=self.abba=None
        self._epoch=stream_epoch; cached=CachedOpticalPage(page,stream_epoch,record_sequence,received)
        if isinstance(page,OpticalLive): self.live=cached
        elif isinstance(page,OpticalAcquisition): self.acquisition=cached
        else: self.abba=cached
        return cached
