import unittest
from unittest.mock import patch
from types import SimpleNamespace
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
class SystemControlTests(unittest.TestCase):
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
 def test_scan_protocol_channels_are_packable_without_relaxing_old_dac_cap(self):
  from ultracoustics._internal.control import pack_manual_request
  for channel in (7,8,9):
   self.assertEqual(pack_manual_request(MANUAL_GET,channel,0,3)[2],channel)
  from ultracoustics import Controller
  c=Controller()
  with self.assertRaisesRegex(ValueError,'44000'):
   c.manual_command(638,MANUAL_SET,CHANNEL_LASER_DAC,50000)
 def test_fp_scan_status_decoder(self):
  f=Fake();f.values[638,9]=(1000<<8)|(1<<4)|4
  self.assertEqual(f.read_fp_scan_638()['state'],'complete')
  self.assertEqual(f.read_fp_scan_638()['count'],1000)
  f.values[638,9]=6
  with self.assertRaisesRegex(RuntimeError,'invalid FP scan status'):f.read_fp_scan_638()
 def test_fp_scan_pairs_feedback_with_preceding_command_and_releases_638(self):
  from ultracoustics._internal.optical_diagnostics import OpticalTrace, OpticalTraceSample
  class ScanFake(Fake):
   def __init__(self):
    super().__init__();self.streaming=True;self.stream_stats={'stream_format':2}
    self.values[1550,2]=1000;self.values[638,7]=50000;self._page_reads=0;self._status_reads=0
    samples=[OpticalTraceSample(288000*i,1000+i,
             round(50000*i/499) if i<500 else 0,0,1) for i in range(501)]
    self._pages=[OpticalTrace(7,i,501,144000000,9,tuple(samples[i:i+2]),42)
                 for i in range(0,501,2)]
   @property
   def telemetry(self):
    self._page_reads+=1
    page=None if self._page_reads==1 else self._pages[min(self._page_reads-2,len(self._pages)-1)]
    cached=None if page is None else SimpleNamespace(page=page,host_stale=False,
                                                     received_monotonic_ns=1<<62)
    return SimpleNamespace(host_stale=False,link_638_stale=False,optical_trace_638=cached)
   def read_fp_scan_638(self,timeout_s=1):
    self._status_reads+=1
    if self._status_reads==1:return {'state':'idle','count':0}
    return {'state':'complete','count':501,'result':'complete'}
  f=ScanFake();clock=[0.0]
  def sleep(seconds):clock[0]+=seconds
  with patch('ultracoustics.system_control.time.monotonic',side_effect=lambda:clock[0]),\
       patch('ultracoustics.system_control.time.sleep',side_effect=sleep):
   result=f.capture_fp_scan(timeout_s=5)
  self.assertEqual(result['paired_count'],500)
  self.assertEqual(result['rows'][0]['commanded_dac'],0)
  self.assertEqual(result['rows'][0]['main_pd_adc_counts'],1001)
  self.assertEqual(result['rows'][-1]['commanded_dac'],50000)
  self.assertEqual(result['rows'][-1]['main_pd_adc_counts'],1500)
  self.assertEqual(f._system_manual_owned,{(1550,2)})
  self.assertIn((638,MANUAL_SET,9,1),f.calls)
  self.assertIn((638,MANUAL_RELEASE,3,0),f.calls)
 def test_close_releases_usb_even_if_stop_is_unconfirmed(self):
  from ultracoustics import Controller
  from unittest.mock import Mock
  c=Controller();c._running=True;c._connected=True;c._stream=Mock()
  stream=c._stream;c.stop_system_confirmed=Mock(side_effect=TimeoutError('USB'))
  with self.assertRaisesRegex(RuntimeError,'physical shutdown was not confirmed'):c.close()
  stream.stop.assert_called_once();stream.close.assert_called_once()
  self.assertFalse(c.connected)
if __name__=='__main__':unittest.main()
