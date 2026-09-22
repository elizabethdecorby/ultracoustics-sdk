import struct,unittest,time
from multiprocessing.shared_memory import SharedMemory
from ultracoustics._internal.telemetry import SIDECAR_BYTES,initialise_sidecar,TelemetrySidecarWriter,read_retained_trace,TelemetrySnapshot
from ultracoustics._internal.optical_diagnostics import parse_page,crc16_ccitt

def raw(index,value=123,capture=1,total=4):
 b=bytearray(52);struct.pack_into('<BBHIHHIHBB',b,0,1,5,52,capture,index,total,144000000,5,2,12)
 for j in range(2):struct.pack_into('<IHHhH',b,20+12*j,1000*(index+j),value,45000,2,1)
 struct.pack_into('<I',b,44,99);struct.pack_into('<H',b,50,crc16_ccitt(b[:50]));return bytes(b)
class RetainedTraceTests(unittest.TestCase):
 def setUp(self):
  self.shm=SharedMemory(create=True,size=SIDECAR_BYTES);initialise_sidecar(self.shm);self.w=TelemetrySidecarWriter(self.shm)
 def tearDown(self):self.shm.close();self.shm.unlink()
 def put(self,b,epoch=1):
  self.w._begin();self.w._retain_trace(parse_page(b),b,epoch);self.w._end()
 def test_slow_reader_keeps_out_of_order_pages_and_duplicates(self):
  self.put(raw(2));self.put(raw(0));self.put(raw(2));r=read_retained_trace(self.shm)
  self.assertTrue(r['complete']);self.assertEqual(len(r['samples']),4)
 def test_missing_and_conflicting_rows_fail_closed(self):
  self.put(raw(0));self.assertEqual(read_retained_trace(self.shm)['missing_indices'],(2,3))
  self.put(raw(2));self.put(raw(0,value=124));r=read_retained_trace(self.shm)
  self.assertTrue(r['conflict']);self.assertFalse(r['complete'])
 def test_new_capture_and_epoch_do_not_mix(self):
  self.put(raw(0));self.put(raw(2,capture=2));self.assertEqual(read_retained_trace(self.shm)['missing_indices'],(0,1))
  self.put(raw(0,capture=2),epoch=2);self.assertEqual(read_retained_trace(self.shm)['missing_indices'],(2,3))
 def test_publish_retains_pages_without_reader_and_epoch_invalidates(self):
  for index in (0,2):
   self.w.publish(TelemetrySnapshot(1,index,time.monotonic_ns(),None,None,optical_page_raw=raw(index)))
  self.assertTrue(read_retained_trace(self.shm)['complete'])
  self.w.publish(TelemetrySnapshot(2,4,time.monotonic_ns(),None,None))
  self.assertIsNone(read_retained_trace(self.shm))
 def test_metadata_change_rejected_and_reset_clears(self):
  self.put(raw(0));self.put(raw(2,total=6));self.assertTrue(read_retained_trace(self.shm)['conflict'])
  self.w.reset_to_legacy();self.assertIsNone(read_retained_trace(self.shm))
if __name__=='__main__':unittest.main()
