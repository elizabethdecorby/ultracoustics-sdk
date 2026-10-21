import struct
import unittest
from unittest.mock import patch
from types import SimpleNamespace
from ultracoustics import identity as ident
from ultracoustics import tec_tuning as tn
from ultracoustics._internal import protocol
from ultracoustics.system_control import SystemControlMixin
from ultracoustics._internal.control import *
class Fake(SystemControlMixin):
 def __init__(self):
  self._system_manual_active=True;self._system_manual_owned={(638,2),(1550,2)};self.calls=[];self._running=True;self.values={}
 def _ensure_connected(self):pass
 def begin_stream(self):pass
 def stop_confirmed(self,timeout_s):
  self.calls.append(('stop',));self._running=False;return {'bulk_write_completed_monotonic_ns':1_000_000_000}
 def runtime_metrics(self,timeout_s):return SimpleNamespace(current_state=0)
 def _send_confirmed(self,*a,**k):self.calls.append(('send',a,k));return {'transport_status':'delivered'}
 def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
  self.calls.append((target,opcode,channel,value))
  if opcode in (MANUAL_SET,MANUAL_TAKE):self.values[target,channel]=value
  if opcode==MANUAL_RELEASE and channel==3:self.values[target,channel]=0
  return SimpleNamespace(target=target,status=0,channel=channel,owner=1,applied_value=self.values.get((target,channel),25000 if channel==1 else 0))
class RecordPIFake(Fake):
 """Contract v2 638 that takes its box-default lock gains from the tuning record (LOCK_TUNING_RECORD.md):
 channel 11 bit 17 always, bit 18 when the record's lock gains are the box defaults, bit 19 when
 they were rejected (profile in use). Channel 11 SET 0 restores the box defaults; anything else BAD_VALUE."""
 PROFILE=(97,450,0)  # belycomm 0.483/449.8 on the wire grid: (kp/0.005, ki 1/s, zero Hz)
 def __init__(self,record=None,rejected=False):
  super().__init__();self._system_manual_active=False;self._system_manual_owned=set()
  self.record,self.rejected=record,rejected
  self.gains,self.over=self.box_defaults(),False
 def box_defaults(self):return self.record if self.record and not self.rejected else self.PROFILE
 def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
  self.calls.append((target,opcode,channel,value));status=0
  if opcode==MANUAL_SET and channel==11:
   if value:status=6
   else:self.gains,self.over=self.box_defaults(),False
  elif opcode==MANUAL_SET and channel==10:
   if value<0 or (value>>30)!=1:status=6
   else:self.gains,self.over=((value>>14)&1023,value&0x3fff,(value>>24)&63),True
  if channel==11:
   used=bool(self.record) and not self.rejected
   raw=2|256|self.over<<9|1024|self.gains[2]<<11|1<<17|used<<18|(bool(self.record) and self.rejected)<<19
  else:raw=self.gains[0]<<14|self.gains[1]
  return SimpleNamespace(target=target,status=status,channel=channel,owner=0,applied_value=raw)
def utid(op,status,record=b'\xff'*128,kind=1):
 return ident.REPLY_MAGIC+struct.pack('<BBHBBBB',ident.REPLY_SCHEMA,op,638,status,255,kind,0)+record
def raw_638_record(payload,seq):
 """A stored 638 tuning record with any payload bytes, bypassing the encoder's checks."""
 header=tn.MAGIC+struct.pack('<BBHI',1,tn.BOARD_638,len(payload),seq)
 raw=header+struct.pack('<I',ident.crc32_mpeg2(header+payload))+payload
 return raw+b'\xff'*(128-len(raw))
class RecordFake(Fake):
 """The mainboard's 'n'/'N' relay (record kind 1) to the 638 bootloader's tuning store, answered
 through the stream (docs/TEC_TUNING.md). The store assigns the sequence. While it handles a
 record the mainboard pauses the ADC stream, and it resumes it ~100 ms after the reply (here:
 after resume_in stream_stats reads); until then it refuses record commands without a reply
 (box 0056, 2026-10-21). mode: None, 'old_master' (identity reply), 'old_boot' (NO_RESPONSE),
 'busy' (BAD_STATE), 'silent' (no reply), 'corrupt' (stores Ki - 1)."""
 def __init__(self,record=None,mode=None):
  super().__init__();self._running=False;self._system_manual_active=False;self._system_manual_owned=set()
  self.record,self.mode,self.requests=record,mode,[]
  self.seq=tn.parse_record(record).seq if record else 0
  self.packets,self.resume_in,self.refused=0,0,0
  self._stream=SimpleNamespace(running=True,_request_stream_control=self.request)
 @property
 def stream_stats(self):
  if self.resume_in:self.resume_in-=1
  else:self.packets+=1
  return {'packets':self.packets}
 def request(self,payload,kind,stream_format=None,timeout_s=1.0,response_timeout_s=None):
  self.requests.append((payload[:5],payload[5:],kind,timeout_s,response_timeout_s))
  op=ident.OP_WRITE if payload[0]==protocol.CMD_IDENTITY_WRITE else ident.OP_READ
  if self.mode=='silent' or self.resume_in:
   self.refused+=bool(self.resume_in)
   raise RuntimeError('stream-control request failed (firmware control response timed out)')
  status={'old_boot':ident.STATUS_NO_RESPONSE,'busy':ident.STATUS_BAD_STATE}.get(self.mode,ident.STATUS_OK)
  if status!=ident.STATUS_BAD_STATE:self.resume_in=3  # still in its record state after the reply
  if status==ident.STATUS_OK and op==ident.OP_WRITE:
   tuning=tn.parse_record(payload[5:]).tuning
   if self.mode=='corrupt':tuning.lock_ki_per_s-=1
   self.seq+=1;self.record=tn.build_record(tuning,seq=self.seq)
  if status==ident.STATUS_OK and self.record is None:status=ident.STATUS_EMPTY
  record=self.record if status==ident.STATUS_OK else b'\xff'*128
  return {'firmware_response':utid(op,status,record,0 if self.mode=='old_master' else 1)}
class SystemControlTests(unittest.TestCase):
 def _manual_readiness_timeout(self,snapshot):
  class Clock:
   ns=5_000_000_000
   def monotonic(self):return self.ns/1e9
   def monotonic_ns(self):return self.ns
   def sleep(self,seconds):self.ns+=int(seconds*1e9)
  clock=Clock()
  class ReadinessFake(Fake):
   def __init__(self):
    super().__init__();self._system_manual_active=False;self._system_manual_owned=set()
    self.snapshot=snapshot;self._stream_stats={'total_samples':120,'fatal':False,
      'ready':False,'device_lost':False,'stream_format':1}
   def stop_confirmed(self,timeout_s):
    self.calls.append(('stop',));self._running=False
    return {'bulk_write_completed_monotonic_ns':1_000_000_000}
   def enable_telemetry(self,timeout_s):pass
   @property
   def telemetry(self):return self.snapshot
   @property
   def stream_stats(self):return self._stream_stats
   def _send_confirmed(self,*args,**kwargs):
    return {'bulk_write_completed_monotonic_ns':clock.monotonic_ns()}
  fake=ReadinessFake()
  with patch('ultracoustics.system_control.time.monotonic',clock.monotonic), \
       patch('ultracoustics.system_control.time.monotonic_ns',clock.monotonic_ns), \
       patch('ultracoustics.system_control.time.sleep',clock.sleep):
   with self.assertRaises(TimeoutError) as caught:fake.begin_system_manual()
  return fake,caught.exception

 def _ready_board(self,tick=42):
  return SimpleNamespace(snapshot_tick_ms=tick,temperature_valid=True,
   temperature_stale=False,temperature_fault=0,temperature_age_ms=2)

 def test_manual_readiness_timeout_identifies_one_missing_board(self):
  snapshot=SimpleNamespace(host_stale=False,host_age_s=0.0,received_monotonic_ns=5_001_000_000,
   record_sequence=9,board_638=None,board_1550=self._ready_board(77),
   link_flags_638=4,link_flags_1550=0)
  fake,error=self._manual_readiness_timeout(snapshot)
  self.assertEqual(str(error),'Manual readiness timed out: 638 link unavailable; 1550 ready')
  self.assertTrue(error.diagnostics['boards'][638]['temperature_valid'] is None)
  self.assertEqual(error.diagnostics['boards'][638]['link_flags'],4)
  self.assertEqual(error.diagnostics['boards'][1550]['snapshot_tick_ms'],77)
  self.assertTrue(error.diagnostics['boards'][1550]['temperature_valid'])
  self.assertFalse(error.diagnostics['snapshot_missing'])
  self.assertFalse(error.diagnostics['snapshot_stale'])
  self.assertEqual(error.diagnostics['adc'],{'samples_total':120,'samples_since_power':0,
   'stream_fatal':False,'stream_ready':False,'device_lost':False,'stream_format':1})
  self.assertFalse(fake.system_manual_active)

 def test_manual_readiness_timeout_reports_whole_snapshot_stale(self):
  snapshot=SimpleNamespace(host_stale=True,host_age_s=.6,received_monotonic_ns=4_400_000_000,
   record_sequence=8,board_638=self._ready_board(),board_1550=self._ready_board(),
   link_flags_638=0,link_flags_1550=0)
  _,error=self._manual_readiness_timeout(snapshot)
  self.assertEqual(str(error),'Manual readiness timed out: ADC telemetry stale')
  self.assertFalse(error.diagnostics['snapshot_missing'])
  self.assertTrue(error.diagnostics['snapshot_stale'])
  self.assertEqual(error.diagnostics['snapshot_record_sequence'],8)
  for target in (638,1550):
   self.assertEqual(error.diagnostics['boards'][target]['temperature_age_ms'],602)
   self.assertLess(error.diagnostics['boards'][target]['post_power_receipt_delay_ms'],0)

 def test_restart_stops_and_observes_dwell(self):
  f=Fake()
  with patch('ultracoustics.system_control.time.monotonic_ns',return_value=1_050_000_000),patch('ultracoustics.system_control.time.sleep') as sleep:
   f.start_confirmed();sleep.assert_called_once_with(.2-.05)
  self.assertEqual(f.calls[0],('stop',));self.assertFalse(f.system_manual_active);self.assertTrue(f._running)
 def test_temperature_takes_without_affecting_other_board(self):
  f=Fake();f.system_manual_command(638,MANUAL_SET,1,25100)
  self.assertIn((638,MANUAL_TAKE,1,25000),f.calls);self.assertIn((1550,2),f._system_manual_owned);self.assertFalse(any(c[0]==1550 for c in f.calls))
 def test_lock_and_dac_ownership_exclusive(self):
  f=Fake()
  with self.assertRaisesRegex(RuntimeError,'manually-owned'):
   f.lock_638()
  self.assertFalse(f.calls)
 def test_automatic_optical_actions_route_without_take_or_stop(self):
  f=Fake();f._system_manual_active=False;f._system_manual_owned=set()
  for action,value in [('reacquire',2),('retune',3),('trace',4),('identify',5)]:
   f.control_638(action)
   self.assertEqual(f.calls[-1],(638,MANUAL_SET,3,value))
  self.assertFalse(any(c[0]=='stop' for c in f.calls))
  for action in ('start','abort'):
   with self.assertRaisesRegex(RuntimeError,'only reacquire, retune, trace, or identify'):
    f.control_638(action)
 def test_queued_retune_rejects_manual_session_before_any_command(self):
  f=Fake();f._system_manual_owned={(638,3),(1550,2)}
  with self.assertRaisesRegex(RuntimeError,'requires automatic locking'):
   f.control_638('retune')
  self.assertFalse(f.calls)
 def test_manual_start_abort_compatibility(self):
  f=Fake();f._system_manual_owned={(638,3),(1550,2)}
  f.lock_638('start');f.lock_638('abort')
  self.assertEqual(f.calls,[(638,MANUAL_SET,3,1),(638,MANUAL_SET,3,0)])
 def test_manual_optical_owner_uses_existing_lease(self):
  f=Fake();f._system_manual_owned={(638,3),(1550,2)}
  f.control_638('reacquire')
  self.assertEqual(f.calls,[(638,MANUAL_SET,3,2)])
 def test_action_capture_reuses_stream_and_bounds_windows(self):
  f=Fake();f._system_manual_active=False;f._system_manual_owned=set();f.streaming=True
  f.save=lambda duration: ('samples',duration)
  f.telemetry=SimpleNamespace(optical_live_638='live',optical_acquisition_638='acq',optical_abba_638='abba')
  with patch('ultracoustics.system_control.time.sleep') as sleep:
   result=f.capture_638_action('reacquire')
  self.assertEqual(result['pre_samples'],('samples',.5));self.assertEqual(result['post_samples'],('samples',.5))
  self.assertEqual(result['acquisition'],'acq');sleep.assert_called_once_with(.5)
  self.assertEqual(f.calls,[(638,MANUAL_SET,3,2)])
 def test_gain_validation_before_writes_and_zero(self):
  f=Fake()
  with self.assertRaises(ValueError):f.optical_pid_638(kp=float('nan'))
  self.assertFalse(f.calls)
  result=f.optical_pid_638(kp=0,ki_negative=.000001,ki_positive=.2)
  self.assertEqual(result['kp'],0);self.assertEqual(result['ki_positive'],.2)
 def test_normalized_pi_atomic_pair_restore_and_capability(self):
  class NormalizedFake(Fake):
   def __init__(self):
    super().__init__();self._system_manual_active=False;self._system_manual_owned=set()
    self.values[638,10]=(200<<14)|900;self.values[638,11]=1|256|1024
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    self.calls.append((target,opcode,channel,value))
    if opcode==MANUAL_SET and channel==10:
     self.values[638,10]=value;self.values[638,11]|=512
    if opcode==MANUAL_SET and channel==11:
     self.values[638,10]=(200<<14)|900;self.values[638,11]&=~512
    raw=self.values.get((target,channel),0)
    if channel==10 and raw & (1<<23):raw-=1<<24
    return SimpleNamespace(target=target,status=0,channel=channel,owner=0,applied_value=raw)
  f=NormalizedFake()
  self.assertEqual(f.read_normalized_pi_638()['ki_per_s'],90)
  result=f.set_normalized_pi_638(.8,95)
  self.assertEqual((result['kp'],result['ki_per_s']),(.8,95))
  self.assertTrue(result['active'] and result['overridden'])
  self.assertIn((638,MANUAL_SET,10,(800<<14)|950),f.calls)
  self.assertFalse(any(c[1]==MANUAL_TAKE for c in f.calls))
  for kp,ki in ((.483,1178.7),(1.0,1638.3),(0.0,0.0)):
   result=f.set_normalized_pi_638(kp,ki)
   self.assertEqual((result['kp'],result['ki_per_s']),(kp,ki))
   self.assertEqual(result['packed'],(round(kp*1000)<<14)|round(ki*10))
  for invalid in (1638.31, -0.1, float('inf'), float('nan')):
   before=len(f.calls)
   with self.assertRaises(ValueError):f.set_normalized_pi_638(.2,invalid)
   # v1 limits need the contract version: at most the info GET, never a SET
   self.assertEqual(f.calls[before:],[(638,MANUAL_GET,11,0)] if invalid==1638.31 else [])
  before=len(f.calls)
  with self.assertRaisesRegex(ValueError,'no second integrator'):f.set_normalized_pi_638(.2,90,zero_hz=20)
  self.assertFalse(any(c[1]==MANUAL_SET for c in f.calls[before:]))
  self.assertEqual(f.read_normalized_pi_638()['zero_hz'],0)
  self.assertFalse(f.restore_normalized_pi_638()['overridden'])
  with self.assertRaises(ValueError):f.set_normalized_pi_638(1.1,95)
  self.assertFalse(any(c[1]==MANUAL_SET and c[2] in (4,5,6) for c in f.calls))
  class OldRangeFake(NormalizedFake):
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    if opcode==MANUAL_SET and channel==10 and (value&0x3fff)>10000:
     self.calls.append((target,opcode,channel,value))
     return SimpleNamespace(target=target,status=6,channel=channel,owner=0,applied_value=0)
    return super().manual_command(target,opcode,channel,value,timeout_s)
  old=OldRangeFake()
  with self.assertRaisesRegex(RuntimeError,'status 6'):old.set_normalized_pi_638(.483,1178.7)
  self.assertEqual(old.read_normalized_pi_638()['ki_per_s'],90)
  self.assertEqual(sum(c[1]==MANUAL_SET for c in old.calls),1)
  class OldFake(NormalizedFake):
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    reply=super().manual_command(target,opcode,channel,value,timeout_s)
    if channel==11:reply.status=4
    return reply
  with self.assertRaisesRegex(RuntimeError,'does not support'):
   OldFake().set_normalized_pi_638(.2,90)
  class OldMasterFake(NormalizedFake):
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    if channel==11:raise RuntimeError('stream-control request failed (USB STALL)')
    return super().manual_command(target,opcode,channel,value,timeout_s)
  with self.assertRaisesRegex(RuntimeError,'capability probe failed; check master/638 firmware support'):
   OldMasterFake().read_normalized_pi_638()
 def test_normalized_pi_contract_v2_word_echo_zero_and_restore(self):
  class V2Fake(Fake):
   """V2_SPEC firmware: 32-bit SET word, 24-bit signed echo, zero_hz in info bits 11-16."""
   def __init__(self):
    super().__init__();self._system_manual_active=False;self._system_manual_owned=set()
    self.kp,self.ki,self.zero,self.over,self.info_extra=.483,449.8,0,False,0
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    self.calls.append((target,opcode,channel,value));status=0
    if opcode==MANUAL_SET and channel==11:self.kp,self.ki,self.zero,self.over=.483,449.8,0,False
    if opcode==MANUAL_SET and channel==10:
     kp5,ki,zero=(value>>14)&1023,value&0x3fff,(value>>24)&63
     if value<0 or (value>>30)!=1:status=6
     else:self.kp,self.ki,self.zero,self.over=kp5*.005,ki,zero,True
    if channel==11:raw=2|256|(512 if self.over else 0)|1024|(self.zero<<11)|self.info_extra
    else:
     raw=(round(self.kp/.005)<<14)|round(self.ki)
     if raw&(1<<23):raw-=1<<24
    return SimpleNamespace(target=target,status=status,channel=channel,owner=0,applied_value=raw)
  f=V2Fake()
  r=f.read_normalized_pi_638()
  self.assertEqual((r['version'],r['kp'],r['ki_per_s'],r['zero_hz'],r['overridden']),(2,.485,450.0,0,False))
  r=f.set_normalized_pi_638(2.6,5000,zero_hz=30)
  self.assertIn((638,MANUAL_SET,10,(1<<30)|(30<<24)|(520<<14)|5000),f.calls)
  self.assertEqual((r['kp'],r['ki_per_s'],r['zero_hz'],r['packed']),(2.6,5000.0,30,(520<<14)|5000))
  self.assertTrue(r['overridden'] and r['active'])
  for kp,ki,zero in ((5.115,16383,63),(0,0,0),(.525,1637.5,20),(1.0,1638.3,0)):
   r=f.set_normalized_pi_638(kp,ki,zero_hz=zero)
   self.assertEqual((r['kp'],r['ki_per_s'],r['zero_hz']),(round(kp/.005)*5/1000,float(round(ki)),zero))
  for args in ((5.12,100),(.5,16383.5),(.5,100,64),(.5,100,1.5),(.5,100,True),(.5,100,-1),(float('nan'),100)):
   before=len(f.calls)
   with self.assertRaises(ValueError):f.set_normalized_pi_638(*args)
   self.assertEqual(len(f.calls),before)
  r=f.restore_normalized_pi_638()
  self.assertEqual((r['overridden'],r['zero_hz'],r['kp'],r['ki_per_s']),(False,0,.485,450.0))
  class DropsZero(V2Fake):
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    reply=super().manual_command(target,opcode,channel,value,timeout_s);self.zero=0;return reply
  with self.assertRaisesRegex(RuntimeError,'verification failed after SET'):
   DropsZero().set_normalized_pi_638(.5,1000,zero_hz=20)
  for extra in (1<<18,1<<19,1<<20,1<<23):  # bits 18-19 need record support (bit 17)
   g=V2Fake();g.info_extra=extra
   with self.assertRaisesRegex(RuntimeError,'unsupported normalized PI contract'):g.read_normalized_pi_638()
  class V3Fake(V2Fake):
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    reply=super().manual_command(target,opcode,channel,value,timeout_s)
    if channel==11:reply.applied_value=(reply.applied_value&~255)|3
    return reply
  with self.assertRaisesRegex(RuntimeError,'unsupported normalized PI contract'):V3Fake().read_normalized_pi_638()
  class V1ExtraBits(V2Fake):
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    reply=super().manual_command(target,opcode,channel,value,timeout_s)
    if channel==11:reply.applied_value=1|256|1024|(20<<11)
    return reply
  with self.assertRaisesRegex(RuntimeError,'unsupported normalized PI contract'):V1ExtraBits().read_normalized_pi_638()
 def _info(self,raw):
  f=Fake();f._system_manual_active=False;f._system_manual_owned=set();f.values[638,11]=raw
  return f._normalized_pi_info_638()
 def test_normalized_pi_record_info_decode(self):
  keys=('record_supported','record_active','record_rejected')
  v2=2|256|1024
  for raw in (1|256|1024,v2,v2|20<<11):  # contract v1, v2 without the record: all False
   self.assertEqual([self._info(raw)[k] for k in keys],[False]*3)
  self.assertEqual([self._info(v2|1<<17)[k] for k in keys],[True,False,False])
  info=self._info(v2|1<<17|1<<18|20<<11)
  self.assertEqual([info[k] for k in keys],[True,True,False])
  self.assertEqual((info['version'],info['active'],info['slope_valid'],info['zero_hz']),(2,True,True,20))
  self.assertEqual([self._info(v2|1<<17|1<<19)[k] for k in keys],[True,False,True])
  for gone in ('storage_supported','stored_active','save_pending','clear_pending','last_write_failed'):
   self.assertNotIn(gone,info)
  for raw in (v2|1<<18,v2|1<<19,v2|1<<17|1<<18|1<<19,v2|1<<17|1<<20,v2|1<<17|1<<21,1|256|1024|1<<17):
   with self.assertRaisesRegex(RuntimeError,'unsupported normalized PI contract'):self._info(raw)
  self.assertFalse(hasattr(Fake(),'store_normalized_pi_638') or hasattr(Fake(),'clear_stored_normalized_pi_638'))
 def test_restore_returns_to_the_record_box_defaults(self):
  f=RecordPIFake(record=(520,5000,30))
  r=f.read_normalized_pi_638()
  self.assertEqual((r['record_active'],r['overridden'],r['kp'],r['ki_per_s'],r['zero_hz']),(True,False,2.6,5000.0,30))
  f.set_normalized_pi_638(.5,1000)
  r=f.restore_normalized_pi_638()  # box defaults are the record's lock gains, second integrator included
  self.assertEqual((r['overridden'],r['record_active'],r['kp'],r['ki_per_s'],r['zero_hz']),(False,True,2.6,5000.0,30))
  self.assertFalse(any(c[1]==MANUAL_SET and c[2]==11 and c[3] for c in f.calls))  # only SET 0
  r=RecordPIFake(record=(520,5000,30),rejected=True).restore_normalized_pi_638()
  self.assertEqual((r['record_active'],r['record_rejected'],r['kp'],r['ki_per_s'],r['zero_hz']),(False,True,.485,450.0,0))
  g=RecordPIFake();g.box_defaults=lambda:(97,450,20)  # a zero without a record is not the profile
  with self.assertRaisesRegex(RuntimeError,'verification failed after restore'):g.restore_normalized_pi_638()
 def test_lock_write_keeps_the_stored_tec_fields_and_returns_the_readback(self):
  tec=tn.TecTuning(board=3,kp=.07,ki=.22,integrator_limit=1.36,setpoint_filter=True,report_id='TUNE_X')
  f=RecordFake(tn.build_record(tec,seq=4))
  with patch('ultracoustics.system_control.time.sleep'):
   r=f.write_lock_tuning_638(.6,1925,report_id='LOCK_0056_DUSTIN')
  (read,_,kind,bound,response_s),(write,record,*_)=f.requests
  self.assertEqual((read,kind,bound,response_s),(protocol.pack_command(protocol.CMD_IDENTITY_READ,1,638),'record',15.5,15.0))
  self.assertEqual((write,len(record),record[6]),(protocol.pack_command(protocol.CMD_IDENTITY_WRITE,1,638),128,60))
  self.assertEqual((r['stored'],r['sequence'],r['rejected']),(True,5,None))
  self.assertEqual((r['lock_kp'],r['lock_ki_per_s'],r['lock_zero_hz'],r['report_id']),(.6,1925.0,0,'LOCK_0056_DUSTIN'))
  self.assertAlmostEqual(r['kp'],.07,places=6);self.assertAlmostEqual(r['integrator_limit'],1.36,places=6)
  self.assertEqual((r['setpoint_filter'],r['conditional_integration'],r['kd']),(True,None,None))
  self.assertEqual(r,f.read_tuning_638())
  self.assertEqual(set(r),{'stored','sequence','rejected','lock_kp','lock_ki_per_s','lock_zero_hz','kp','ki','kd',
                           'integrator_limit','setpoint_filter','conditional_integration','tuned_at','report_id'})
 def test_record_exchanges_wait_for_the_mainboard_to_resume_streaming(self):
  # Box 0056 (master 9514d0a), 2026-10-21: a read sent right after another reply got no reply (the
  # mainboard was still in its record state); after the ADC stream resumed (0.10 s) it was answered.
  f=RecordFake()
  r1,r2=f.read_tuning_638(),f.read_tuning_638()  # back to back, as record_test.py's read + write did
  r3=f.write_lock_tuning_638(.575,1575)          # its read, then its write
  self.assertEqual((f.refused,len(f.requests),r1,r2['stored'],r3['lock_kp']),(0,4,r2,False,.575))
  self.assertEqual(f.resume_in,0)  # back in IDLE on return, so a START is not ignored
  f.resume_in=5  # an earlier maintenance command (e.g. a stream format change) not finished yet
  self.assertEqual(f.read_tuning_638()['sequence'],1);self.assertEqual(f.refused,0)
  class OldSDK(RecordFake):
   def _await_mainboard_streaming(self,limit_s=1.0):return True
  g=OldSDK();g.read_tuning_638()
  with self.assertRaisesRegex(RuntimeError,'no reply; the mainboard refused'):g.read_tuning_638()
  self.assertEqual(g.refused,1)
  class Frozen(RecordFake):  # no ADC packets at all: give up after the bound and send anyway
   stream_stats={'packets':7}
  clock=[0.0]
  with patch('ultracoustics.system_control.time.monotonic',side_effect=lambda:clock[0]), \
       patch('ultracoustics.system_control.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
   self.assertFalse(Frozen()._await_mainboard_streaming())
   self.assertFalse(Frozen().read_tuning_638()['stored'])
  self.assertGreaterEqual(clock[0],3.0)
 def test_lock_clear_keeps_tec_fields_and_an_empty_board_reads_unset(self):
  f=RecordFake()
  r=f.read_tuning_638()
  self.assertEqual((r['stored'],r['sequence'],r['lock_kp'],r['kp'],r['rejected']),(False,None,None,None,None))
  self.assertEqual(f.requests[0][4],8.0)
  with patch('ultracoustics.system_control.time.sleep'):
   r=f.write_lock_tuning_638(2.6,5000,zero_hz=30)
   self.assertEqual((r['lock_kp'],r['lock_ki_per_s'],r['lock_zero_hz'],r['kp'],r['sequence']),(2.6,5000.0,30,None,1))
   f.record=tn.build_record(tn.TecTuning(board=3,kp=.08,lock_kp=2.6,lock_ki_per_s=5000.0,lock_zero_hz=30),seq=7);f.seq=7
   r=f.clear_lock_tuning_638()
  self.assertEqual((r['stored'],r['sequence'],r['lock_kp'],r['lock_ki_per_s'],r['lock_zero_hz']),(True,8,None,None,None))
  self.assertAlmostEqual(r['kp'],.08,places=6)
  self.assertEqual(f.record[6],48)  # no lock field left: a layout-v1 48-byte payload
 def test_tuning_record_needs_idle_a_stream_no_manual_session_and_wire_grid_gains(self):
  f=RecordFake();f._running=True
  calls=(f.read_tuning_638,lambda:f.write_lock_tuning_638(.6,1925),f.clear_lock_tuning_638)
  for call in calls:
   with self.assertRaisesRegex(RuntimeError,'IDLE; STOP first'):call()
  f._running=False;f._system_manual_active=True
  for call in calls:
   with self.assertRaisesRegex(RuntimeError,'Finish the manual session'):call()
  f._system_manual_active=False;f._stream.running=False
  for call in calls:
   with self.assertRaisesRegex(RuntimeError,'begin_stream'):call()
  f._stream.running=True
  for args in ((.601,1925),(.6,1925.5),(5.12,1),(.6,16384),(-.005,1),(float('nan'),1),('0.6',1),(True,1),
               (.6,1925,64),(.6,1925,True),(.6,1925,1.0),(.6,1925,-1)):
   with self.assertRaises(ValueError):f.write_lock_tuning_638(*args)
  with self.assertRaises(ValueError):f.write_lock_tuning_638(.6,1925,report_id='x'*25)
  self.assertEqual(f.requests,[])
  from ultracoustics import Controller
  with self.assertRaisesRegex(RuntimeError,'begin_stream'):Controller().read_tuning_638()
 def test_tuning_record_without_support_or_with_failures_raises_and_writes_nothing(self):
  for mode,message in (('old_master','does not support TEC tuning records'),
                       ('old_boot','bootloader without tuning-record support'),
                       ('busy','not in IDLE or OVERRIDE'),('silent','no reply; the mainboard refused')):
   f=RecordFake(mode=mode)
   with self.assertRaisesRegex(RuntimeError,message):f.read_tuning_638()
   with patch('ultracoustics.system_control.time.sleep'):
    for call in (lambda:f.write_lock_tuning_638(.6,1925),f.clear_lock_tuning_638):
     with self.assertRaisesRegex(RuntimeError,message):call()
   self.assertEqual([q[0][0] for q in f.requests],[protocol.CMD_IDENTITY_READ]*3)  # the read failed: no write
  f=RecordFake(mode='corrupt')
  with patch('ultracoustics.system_control.time.sleep'):
   with self.assertRaisesRegex(RuntimeError,'VERIFY_FAIL'):f.write_lock_tuning_638(.6,1925)
  class Mismatch(RecordFake):
   def request(self,*a,**k):
    reply=super().request(*a,**k);reply['firmware_response']=utid(ident.OP_WRITE,ident.STATUS_EMPTY);return reply
  with self.assertRaisesRegex(RuntimeError,'does not match the request'):Mismatch().read_tuning_638()
 def test_rejected_groups_are_reported_and_a_rejected_tec_group_is_not_kept(self):
  lock=struct.pack('<ff',.6,1925.0)+b'\x00\xff\xff\xff'
  f=RecordFake(raw_638_record(struct.pack('<f',90.0)+b'\xff'*44+lock,3))
  r=f.read_tuning_638()
  self.assertEqual((r['kp'],r['lock_kp'],r['lock_ki_per_s']),(90.0,.6,1925.0))
  self.assertEqual(r['rejected'],'TEC fields rejected: kp = 90.0 is outside [0, 1]')
  with patch('ultracoustics.system_control.time.sleep'):
   r=f.write_lock_tuning_638(.65,2000)
  self.assertEqual((r['kp'],r['lock_kp'],r['lock_ki_per_s'],r['rejected']),(None,.65,2000.0,None))
  r=RecordFake(raw_638_record(b'\xff'*48+struct.pack('<ff',6.0,1925.0)+b'\x40\xff\xff\xff',3)).read_tuning_638()
  self.assertEqual(r['rejected'],'lock fields rejected: lock_kp = 6.0 is outside [0, 5.115]')
  r=RecordFake(raw_638_record(struct.pack('<f',90.0)+b'\xff'*52+b'\x40',3)).read_tuning_638()
  self.assertEqual(r['rejected'],'TEC fields rejected: kp = 90.0 is outside [0, 1]; '
                                 'lock fields rejected: lock_zero_hz = 64 is outside 0..63')
 def test_renew_failure_stops_both(self):
  f=Fake()
  def fail(*a,**k):raise TimeoutError('ack')
  f.manual_command=fail
  with self.assertRaisesRegex(RuntimeError,'system stopped'):f.renew_system_manual()
  self.assertFalse(f.system_manual_active);self.assertEqual(f.calls,[('stop',)])
 def test_read_pid_never_changes_dac_or_ownership(self):
  f=Fake();f.optical_pid_638()
  self.assertTrue(all(c[1]==MANUAL_GET for c in f.calls))
  self.assertEqual(f._system_manual_owned,{(638,2),(1550,2)})
 def test_new_638_cap_and_runtime_word(self):
  f=Fake();f.values[638,7]=50000;f.values[638,8]=(3<<8)|1
  self.assertEqual(f.read_optical_cap_638()['max_dac'],50000)
  self.assertEqual(f.read_controller_timing_638(),{'divider':3,'held':True,'raw':769})
  f._system_manual_owned={(638,3),(1550,2)}
  self.assertEqual(f.set_controller_timing_638(10,False)['divider'],10)
  self.assertEqual(f.calls[-1],(638,MANUAL_SET,8,10<<8))
  with self.assertRaises(ValueError):f.set_controller_timing_638(4,False)
  f._system_manual_active=False;f._system_manual_owned=set()
  self.assertEqual(f.set_controller_timing_638(1,True)['raw'],257)
  self.assertEqual(f.calls[-1],(638,MANUAL_SET,8,257))
 def test_scan_requires_illumination_before_handoff(self):
  f=Fake();f.streaming=True;f.stream_stats={'stream_format':2}
  with self.assertRaisesRegex(RuntimeError,'nonzero 1550'):
   f.capture_fp_scan(timeout_s=2)
  self.assertEqual(f._system_manual_owned,{(638,2),(1550,2)})
  self.assertFalse(any(c[1]==MANUAL_SET and c[2]==9 for c in f.calls))
 def test_scan_timeout_bounds(self):
  f=Fake()
  for timeout in (.5,75.1):
   with self.assertRaisesRegex(ValueError,'between 1 and 75'):
    f.capture_fp_scan(timeout_s=timeout)
  self.assertFalse(f.calls)
  f.streaming=True;f.stream_stats={'stream_format':2}
  with self.assertRaisesRegex(RuntimeError,'nonzero 1550'):
   f.capture_fp_scan(timeout_s=75)
 def test_scan_protocol_channels_are_packable_with_the_manual_dac_cap_enforced(self):
  from ultracoustics._internal.control import pack_manual_request
  for channel in (7,8,9):
   self.assertEqual(pack_manual_request(MANUAL_GET,channel,0,3)[2],channel)
  from ultracoustics import Controller
  c=Controller()
  with self.assertRaisesRegex(ValueError,'52400'):
   c.manual_command(638,MANUAL_SET,CHANNEL_LASER_DAC,52401)
 def test_fp_scan_status_decoder(self):
  f=Fake();f.values[638,9]=(1000<<8)|(1<<4)|4
  self.assertEqual(f.read_fp_scan_638()['state'],'complete')
  self.assertEqual(f.read_fp_scan_638()['count'],1000)
  f.values[638,9]=6
  with self.assertRaisesRegex(RuntimeError,'invalid FP scan status'):f.read_fp_scan_638()
 def test_live_scan_capture_alignment_and_loss_rejection(self):
  import struct
  import numpy as np
  from ultracoustics.system_control import FPScanError, SlaveCommandRejected
  clock=[0.0]
  class LiveFake(Fake):
   def __init__(self,fail=False,arm_busy=0,ambiguous=False):
    super().__init__();self.streaming=True;self.fail=fail
    self.arm_busy=arm_busy;self.ambiguous=ambiguous;self.arm_attempts=0
    self.scan_origin=None;self.scan_started=None
    self.values[1550,2]=1000;self.values[638,7]=44000
    self.buffer_capacity=20_000_000;self.buffer=np.full(self.buffer_capacity,1234,dtype=np.uint16)
   def manual_command(self,target,opcode,channel,value=0,timeout_s=1):
    if target==638 and opcode==MANUAL_SET and channel==9 and value==1:
     self.arm_attempts+=1
     if self.ambiguous:raise TimeoutError('ambiguous command response')
     if self.arm_busy:
      self.arm_busy-=1
      reply=SimpleNamespace(target=638,transaction=self.arm_attempts,opcode=opcode,
       channel=channel,status=10,owner=1,applied_value=0,raw_ack=b'busy-ack')
      raise SlaveCommandRejected(target,channel,10,opcode,reply)
     self.scan_origin=self.samples_received;self.scan_started=clock[0]
    return super().manual_command(target,opcode,channel,value,timeout_s)
   @property
   def samples_received(self):return (int(clock[0]*10_000_000)//8192+1)*8192
   @property
   def stream_stats(self):return {'stream_format':2,'drops_seq':int(self.fail and clock[0]>.4)}
   @property
   def telemetry(self):
    total=self.samples_received
    data=None
    if self.scan_origin is not None and clock[0]-self.scan_started>.003:
     data=bytearray(54);data[0]=1;data[1]=3 if clock[0]-self.scan_started>1.01 else 1
     struct.pack_into('<I',data,4,1)
     for off,point in zip((8,14,20,26),
       (self.scan_origin+20000,self.scan_origin+21000,
        self.scan_origin+10020000,self.scan_origin+10021000)):
      struct.pack_into('<IH',data,off,point//8192,point%8192)
    return SimpleNamespace(host_stale=False,stream_epoch=1,scan_sync=data,
                           record_sequence=total//8192-1,received_sample_end=total)
   def read_fp_scan_638(self,timeout_s=1):return {'state':'complete','result':'complete','count':1001}
  for cap,fail in ((44000,False),(52400,False),(52400,True)):
   clock[0]=0;f=LiveFake(fail);f.values[638,7]=cap;updates=[]
   with patch('ultracoustics.system_control.time.monotonic',side_effect=lambda:clock[0]),patch('ultracoustics.system_control.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
    if fail:
     with self.assertRaisesRegex(FPScanError,'discontinuity'):f.capture_fp_scan()
     self.assertIn((638,MANUAL_SET,9,0),f.calls)
    else:
     result=f.capture_fp_scan(on_progress=updates.append)
     self.assertEqual(len(result['raw_adc']),10_000_000)
     self.assertEqual(result['rows'][0]['main_pd_adc_counts'],1234)
     self.assertEqual(result['rows'][-1]['commanded_dac'],cap)
     self.assertEqual(result['alignment_bound_us'],56.6)
     self.assertTrue(any(u['new_rows'] for u in updates))
     self.assertLess(clock[0],1.5)
   self.assertIn((638,MANUAL_RELEASE,3,0),f.calls)

  clock[0]=0;f=LiveFake(arm_busy=3);updates=[]
  with patch('ultracoustics.system_control.time.monotonic',side_effect=lambda:clock[0]), \
       patch('ultracoustics.system_control.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
   result=f.capture_fp_scan(on_progress=updates.append)
  self.assertEqual(f.arm_attempts,4)
  self.assertEqual(len(result['arm_rejections']),3)
  self.assertEqual(result['arm_rejections'][0]['raw_ack_hex'],b'busy-ack'.hex())
  self.assertTrue(any(update['state']=='Waiting for 638 readiness' for update in updates))
  self.assertEqual(len(result['raw_adc']),10_000_000)

  clock[0]=0;f=LiveFake(arm_busy=1000);f.values[638,7]=52400
  f.buffer_capacity=100_000;f.buffer=np.full(f.buffer_capacity,1234,dtype=np.uint16)
  with patch('ultracoustics.system_control.time.monotonic',side_effect=lambda:clock[0]), \
       patch('ultracoustics.system_control.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
   with self.assertRaises(FPScanError) as caught:f.capture_fp_scan(timeout_s=2.0)
  self.assertLessEqual(clock[0],2.0)
  self.assertEqual(f.arm_attempts,len(caught.exception.arm_rejections))
  self.assertLess(f.arm_attempts,50)
  self.assertIn((638,MANUAL_RENEW,3,0),f.calls)
  self.assertIn((1550,MANUAL_RENEW,2,0),f.calls)

  clock[0]=0;f=LiveFake(ambiguous=True);f.values[638,7]=52400
  with patch('ultracoustics.system_control.time.monotonic',side_effect=lambda:clock[0]), \
       patch('ultracoustics.system_control.time.sleep',side_effect=lambda dt:clock.__setitem__(0,clock[0]+dt)):
   with self.assertRaisesRegex(FPScanError,'ambiguous command response'):
    f.capture_fp_scan()
  self.assertEqual(f.arm_attempts,1)
 def test_close_releases_usb_even_if_stop_is_unconfirmed(self):
  from ultracoustics import Controller
  from unittest.mock import Mock
  c=Controller();c._running=True;c._connected=True;c._stream=Mock()
  stream=c._stream;c.stop_system_confirmed=Mock(side_effect=TimeoutError('USB'))
  with self.assertRaisesRegex(RuntimeError,'physical shutdown was not confirmed'):c.close()
  stream.stop.assert_called_once();stream.close.assert_called_once()
  self.assertFalse(c.connected)
if __name__=='__main__':unittest.main()
