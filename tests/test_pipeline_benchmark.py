"""Benchmark validation and correctness comparisons without loading ML models."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch

from tools.benchmark_pipeline import output_path,validate_device,equivalent,parser,ROOT,main


def trial():
    return {'trace':[{'frame_index':0,'detections':[[1.,2.,3.,4.,.9]],'tracks':[[1.,2.,3.,4.,1]]}],
            'events':[{'track':1,'frame':0,'point':[2.,4.],'timestamp_ms':10.,'signal':'red','status':'candidate'}],
            'accepted_plates':0}


class BenchmarkTests(unittest.TestCase):
    def test_cpu_only_device_without_cuda_probe(self):
        torch=Mock();self.assertEqual(validate_device('cpu',torch),'cpu')
        torch.cuda.is_available.assert_not_called()

    def test_auto_preserves_existing_device_selection(self):
        self.assertIsNone(validate_device('auto',Mock()))

    def test_available_cuda_index(self):
        torch=Mock();torch.cuda.is_available.return_value=True;torch.cuda.device_count.return_value=1
        self.assertEqual(validate_device('0',torch),'0')
        with self.assertRaises(ValueError):validate_device('1',torch)

    def test_unavailable_gpu_does_not_fall_back_silently(self):
        torch=Mock();torch.cuda.is_available.return_value=False
        with self.assertRaises(ValueError):validate_device('0',torch)

    def test_arbitrary_device_input_rejected(self):
        for value in ('cuda:0','mps','nan','-1','０'):
            with self.subTest(value=value),self.assertRaises(ValueError):validate_device(value,Mock())

    def test_report_must_stay_inside_ignored_outputs(self):
        for value in ('report.json','outputs/../main.json','../outside.json','outputs/report.txt'):
            with self.subTest(value=value),self.assertRaises(ValueError):output_path(value)

    def test_report_existing_file_not_overwritten(self):
        with tempfile.NamedTemporaryFile(dir=str(ROOT/'outputs'),suffix='.json') as handle:
            with self.assertRaises(ValueError):output_path(handle.name)

    def test_output_validation_does_not_create_directories_or_files(self):
        path=output_path('outputs/phase8-test-unused/new.json')
        self.assertFalse(path.exists());self.assertFalse(path.parent.exists())

    def test_csv_output_is_optional_and_validated(self):
        self.assertEqual(output_path('outputs/phase8-unused.csv','.csv').suffix,'.csv')
        with self.assertRaises(ValueError):output_path('outputs/phase8-unused.json','.csv')

    def test_equivalent_trace_is_accepted(self):
        self.assertTrue(equivalent(trial(),deepcopy(trial())))

    def test_same_candidate_count_does_not_hide_missing_detections(self):
        left=trial();right=deepcopy(left);right['trace'][0]['detections']=[]
        self.assertFalse(equivalent(left,right))

    def test_track_identity_and_observation_changes_rejected(self):
        left=trial();right=deepcopy(left);right['trace'][0]['tracks'][0][4]=2
        self.assertFalse(equivalent(left,right))

    def test_crossing_frame_time_signal_and_point_changes_rejected(self):
        for key,value in [('frame',1),('timestamp_ms',11),('signal','green'),('point',[3.,4.])]:
            left=trial();right=deepcopy(left);right['events'][0][key]=value
            self.assertFalse(equivalent(left,right))

    def test_documented_detection_float_tolerance_only(self):
        left=trial();right=deepcopy(left);right['trace'][0]['detections'][0][0]+=1e-5
        self.assertTrue(equivalent(left,right))
        right['trace'][0]['detections'][0][0]+=.001
        self.assertFalse(equivalent(left,right))

    def test_invalid_frame_warmup_and_trial_values_stop_before_models(self):
        for option,value in [('--frames','0'),('--frames','6001'),('--warmup','-1'),('--trials','0'),('--viewers','9'),('--cpu-threads','0'),('--timeout-seconds','0')]:
            with self.subTest(option=option),patch('tools.benchmark_pipeline.run_trial') as run,self.assertRaises(SystemExit):
                main([option,value,'--worker','--output','outputs/phase8-unused.json'])
            run.assert_not_called()

    def test_default_benchmark_is_explicit_cpu_unpaced_workload(self):
        args=parser().parse_args(['--output','outputs/test.json'])
        self.assertEqual(args.frames,60);self.assertEqual(args.device,'cpu')
        self.assertEqual(args.trials,3);self.assertEqual(args.viewers,1)
        self.assertFalse(args.no_storage);self.assertFalse(args.evidence)


if __name__=='__main__':unittest.main()
