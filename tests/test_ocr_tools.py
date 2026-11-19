"""Known-label OCR metrics and offline benchmark contracts, with fake readers."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from tools.benchmark_ocr import benchmark, load_images, main as benchmark_main
from tools.evaluate_ocr import edit_distance, evaluate, load_manifest, main, write_report
from tools.detection_common import PROJECT_ROOT


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels=[{'id':'one','text':'UP15AB1234','format':'standard'},
                     {'id':'two','text':'22BH1234AA','format':'bharat'},
                     {'id':'negative','text':'','format':'negative'}]

    def predictions(self):
        return [{'id':s['id'],'strategies':{'fixed':{'text':s['text'],'accepted':bool(s['text'])}}} for s in self.labels]

    def test_perfect_positive_and_negative(self):
        row=evaluate(self.labels,self.predictions())['metrics'][0]
        self.assertEqual(row['exact_match_accuracy'],1)
        self.assertEqual(row['character_error_rate'],0)
        self.assertEqual(row['false_acceptance_rate'],0)

    def test_character_error_rate_includes_rejections(self):
        pred=self.predictions()
        pred[0]['strategies']['fixed'].update(accepted=False,text='')
        row=evaluate(self.labels,pred)['metrics'][0]
        self.assertEqual(row['exact_match_accuracy'],.5)
        self.assertEqual(row['character_error_rate'],.5)

    def test_false_acceptance_denominator(self):
        pred=self.predictions()
        pred[2]['strategies']['fixed'].update(accepted=True,text='MH12AB1234')
        row=evaluate(self.labels,pred)['metrics'][0]
        self.assertAlmostEqual(row['false_acceptance_rate'],1/3)
        self.assertAlmostEqual(row['false_acceptance_per_sample'],1/3)

    def test_no_acceptance_denominator_is_null(self):
        pred=self.predictions()
        for sample in pred:
            sample['strategies']['fixed'].update(accepted=False,text='')
        self.assertIsNone(evaluate(self.labels,pred)['metrics'][0]['false_acceptance_rate'])

    def test_per_format_and_stage_rates_not_inferred(self):
        report=evaluate(self.labels,self.predictions())
        self.assertEqual(len(report['metrics']),4)
        self.assertIsNone(report['stage_observation_rates']['plate_detected'])

    def test_stage_rates_require_explicit_booleans(self):
        pred=self.predictions()
        pred[0]['plate_detected']=True
        pred[1]['plate_detected']=False
        self.assertEqual(evaluate(self.labels,pred)['stage_observation_rates']['plate_detected'],.5)

    def test_missing_or_duplicate_predictions_fail(self):
        for pred in (self.predictions()[:1],self.predictions()+self.predictions()[:1]):
            with self.assertRaises(ValueError):
                evaluate(self.labels,pred)

    def test_incomplete_strategy_coverage_fails(self):
        pred=self.predictions()
        pred[0]['strategies']['gray']={'text':'','accepted':False}
        with self.assertRaises(ValueError):
            evaluate(self.labels,pred)

    def test_invalid_prediction_fails(self):
        for result in ({'text':'','accepted':True},{'text':'UP15AB1234','accepted':'yes'}):
            pred=self.predictions()
            pred[0]['strategies']['fixed']=result
            with self.assertRaises(ValueError):
                evaluate(self.labels,pred)

    def test_edit_distance_fixture(self):
        self.assertEqual(edit_distance('UP15AB1234','UP15AB123X'),1)
        self.assertEqual(edit_distance('ABC',''),3)

    def test_manifest_validation(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=Path(scratch)/'manifest.json'
            for value in ({'schema_version':1,'samples':[]}, {'schema_version':1,'samples':[{'id':'x'},{'id':'x'}]}):
                path.write_text(json.dumps(value))
                with self.assertRaises(ValueError):
                    load_manifest(path)

    def test_json_csv_cli_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs',prefix='ocr-test-') as scratch:
            root=Path(scratch)
            labels=root/'labels.json';pred=root/'pred.json'
            labels.write_text(json.dumps({'schema_version':1,'samples':self.labels}))
            pred.write_text(json.dumps({'schema_version':1,'samples':self.predictions()}))
            for suffix in ('json','csv'):
                target=root/('report.'+suffix)
                self.assertEqual(main(['--annotations',str(labels),'--predictions',str(pred),'--output',str(target)]),0)
                self.assertTrue(target.stat().st_size)
                with self.assertRaises(ValueError):
                    write_report(target,{})
            with (root/'report.csv').open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))),4)

    def test_reports_must_remain_under_outputs(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(ValueError):
                write_report(Path(scratch)/'report.json',{})


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.reader=Mock()
        self.reader.device='cpu'
        self.reader.readtext.return_value=[(None,'UP15AB1234',.95)]
        self.samples=[{'id':'one'}]
        self.image=np.zeros((20,40,3),dtype=np.uint8)
        self.images=[(self.image,'hash')]

    def test_identical_crops_warmup_and_call_counts(self):
        report=benchmark(self.reader,self.samples,self.images,('color','fixed'),2,1,processed_frames=10,violation_candidates=2)
        self.assertEqual(self.reader.readtext.call_count,6)
        self.assertEqual(report['runs'][0]['calls'],2)
        self.assertEqual(report['runs'][0]['ocr_calls_per_frame'],.1)
        self.assertEqual(report['runs'][0]['ocr_calls_per_candidate'],.5)
        self.assertFalse(self.image.any())

    def test_timing_units_and_device(self):
        result=benchmark(self.reader,self.samples,self.images,('fixed',),1,0)
        self.assertEqual(result['hardware']['device'],'cpu')
        self.assertEqual(set(result['runs'][0]['ocr_latency']),{'mean_ms','median_ms','p95_ms'})
        self.assertGreater(result['runs'][0]['crop_pipeline_fps'],0)

    def test_without_labels_no_accuracy_claim(self):
        result=benchmark(self.reader,self.samples,self.images,('fixed',),1,0)
        self.assertNotIn('metrics',result)
        self.assertIsNone(result['runs'][0]['ocr_calls_per_frame'])

    def test_with_labels_exact_match(self):
        self.samples[0].update(text='UP15AB1234',format='standard')
        result=benchmark(self.reader,self.samples,self.images,('fixed',),1,0)
        self.assertEqual(result['metrics'][0]['exact_match_accuracy'],1)

    def test_invalid_options(self):
        for options in ({'repeats':0},{'warmup':-1},{'methods':['noise']},{'processed_frames':0}):
            with self.assertRaises(ValueError):
                benchmark(self.reader,self.samples,self.images,**options)
        self.reader.readtext.assert_not_called()

    def test_missing_and_empty_images(self):
        with tempfile.TemporaryDirectory() as scratch:
            with self.assertRaises(ValueError):
                load_images([{'id':'x','image':'absent.png'}],Path(scratch)/'manifest.json')
            path=Path(scratch)/'empty.png'
            path.write_bytes(b'')
            with self.assertRaises(ValueError):
                load_images([{'id':'x','image':'empty.png'}],Path(scratch)/'manifest.json')

    def test_load_image_without_changing_bytes(self):
        with tempfile.TemporaryDirectory() as scratch:
            path=Path(scratch)/'plate.png'
            cv2.imwrite(str(path),self.image)
            before=path.read_bytes()
            images=load_images([{'id':'x','image':'plate.png'}],Path(scratch)/'manifest.json')
            self.assertEqual(path.read_bytes(),before)
            np.testing.assert_array_equal(images[0][0],self.image)

    def test_benchmark_cli_uses_mocked_offline_reader(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs',prefix='ocr-cli-') as scratch:
            root=Path(scratch)
            cv2.imwrite(str(root/'plate.png'),self.image)
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'schema_version':1,'samples':[{'id':'one','image':'plate.png'}]}))
            with patch('util.get_reader',return_value=self.reader):
                self.assertEqual(benchmark_main(['--manifest',str(manifest),'--methods','fixed',
                    '--warmup','0','--output',str(root/'report.json')]),0)
            self.assertEqual(json.loads((root/'report.json').read_text())['runs'][0]['calls'],1)

    def test_bad_cli_options_fail_before_reader_initialization(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT/'outputs',prefix='ocr-cli-') as scratch:
            root=Path(scratch)
            cv2.imwrite(str(root/'plate.png'),self.image)
            manifest=root/'manifest.json'
            manifest.write_text(json.dumps({'schema_version':1,'samples':[{'id':'one','image':'plate.png'}]}))
            with patch('util.get_reader') as reader,self.assertRaises(SystemExit):
                benchmark_main(['--manifest',str(manifest),'--repeats','0','--output',str(root/'report.json')])
            reader.assert_not_called()
            self.assertFalse((root/'report.json').exists())


if __name__=='__main__':
    unittest.main()
