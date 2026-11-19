"""Model-free OCR rules, association, bounded evidence and application regressions."""
from dataclasses import replace
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from plate_recognition import (OCRConfig, OCRManager, associate_plate, coerce_reading,
                               correction_candidates, normalize_text, preprocess,
                               probability, rank_ocr, safe_crop, validate_plate)


def reading(text='UP15AB1234', score=.8, config=None):
    return rank_ocr([(None,text,score)],config)


def event(identity=1):
    return {'event_id':'event-'+str(identity),'track_id':identity,'license_plate':None,
            'license_plate_confidence':None,'status':'candidate'}


class FormatTests(unittest.TestCase):
    def test_standard_examples_and_variable_lengths(self):
        for text in ('UP15AB1234','KA03MN1234','DL8CAF5032','MH12AB1234','TN1A1234'):
            with self.subTest(text=text):
                self.assertEqual(validate_plate(text).format_type,'standard')

    def test_bharat_series(self):
        self.assertEqual(validate_plate('22BH1234AA').format_type,'bharat')

    def test_unsupported_and_arbitrary_noise(self):
        for text in ('ZZ15AB1234','UP123AB1234','UP15AB123','1234567890','HELLOWORLD'):
            self.assertFalse(validate_plate(text).is_valid)

    def test_legacy_can_be_disabled(self):
        self.assertEqual(validate_plate('AB12CDE').format_type,'legacy')
        self.assertFalse(validate_plate('AB12CDE',replace(OCRConfig(),allow_legacy=False)).is_valid)

    def test_custom_pattern(self):
        config=replace(OCRConfig(),extra_patterns=(('deployment',r'TEST[0-9]{3}'),))
        self.assertEqual(validate_plate('TEST123',config).format_type,'deployment')
        config=replace(OCRConfig(),extra_patterns=(('deployment',r'ZZ[0-9]{2}[A-Z]{2}[0-9]{4}'),))
        self.assertFalse(correction_candidates('ZZI5AB1234',.9,config))

    def test_spaces_hyphens_and_case(self):
        self.assertEqual(normalize_text('  up-15 ab 1234\t'),'UP15AB1234')

    def test_punctuation_not_erased(self):
        self.assertEqual(normalize_text('UP15AB12?4'),'UP15AB12?4')
        self.assertFalse(validate_plate('UP15AB12?4').is_valid)

    def test_empty_nonstring_unicode_and_long_text(self):
        for value in ('','   ',None,123,[], 'ßP15AB1234','UP15AB1234'+' '*65):
            self.assertEqual(normalize_text(value),'')

    def test_numeric_position_correction(self):
        result=correction_candidates('UPI5ABC1234',.8)
        self.assertEqual([c.normalized_text for c in result],['UP15ABC1234'])
        self.assertEqual(result[0].original_text,'UPI5ABC1234')
        self.assertEqual(result[0].corrections,1)

    def test_alphabetic_position_correction(self):
        self.assertEqual(correction_candidates('UP15A81234',.8)[0].normalized_text,'UP15AB1234')

    def test_valid_unmodified_text_never_substituted(self):
        candidate=correction_candidates('UP15BO1234',.99)[0]
        self.assertEqual(candidate.normalized_text,'UP15BO1234')
        self.assertEqual(candidate.corrections,0)

    def test_corrections_bounded_and_invalid_rejected(self):
        self.assertFalse(correction_candidates('UPI5A8I234',.9))
        self.assertFalse(correction_candidates('UP15A?1234',.9))
        self.assertFalse(correction_candidates('UPI5AB1234',.9,replace(OCRConfig(),max_corrections=0)))

    def test_ambiguous_positions_generate_bounded_candidates(self):
        result=correction_candidates('UPI5AB1234',.9)
        self.assertEqual({c.normalized_text for c in result},{'UP15AB1234','UP1SAB1234'})
        self.assertEqual(reading('UPI5AB1234',.9).status,'conflicting')

    def test_bharat_literal_prefix_not_guessed(self):
        self.assertFalse(correction_candidates('228H1234AA',.9))

    def test_candidate_cap_cannot_hide_ambiguity(self):
        self.assertFalse(correction_candidates('UPI5AB1234',.9,replace(OCRConfig(),max_candidates=1)))


class RankingTests(unittest.TestCase):
    def test_threshold_boundary(self):
        self.assertIsNotNone(reading(score=.65).best)
        self.assertEqual(reading(score=.64).status,'low_confidence')

    def test_invalid_confidence(self):
        for value in (float('nan'),float('inf'),-1,1.1,True,'0.9',None):
            self.assertFalse(probability(value))
            self.assertIsNone(reading(score=value).best)

    def test_highest_of_multiple_results(self):
        result=rank_ocr([(None,'UP15AB1234',.7),(None,'KA03MN1234',.9)])
        self.assertEqual(result.best.normalized_text,'KA03MN1234')

    def test_valid_unmodified_precedes_speculative(self):
        result=rank_ocr([(None,'UPI5AB1234',.99),(None,'KA03MN1234',.8)])
        self.assertEqual(result.best.normalized_text,'KA03MN1234')

    def test_competing_close_results_conflict(self):
        result=rank_ocr([(None,'UP15AB1234',.9),(None,'KA03MN1234',.89)])
        self.assertEqual(result.status,'conflicting')
        self.assertIsNone(result.best)

    def test_empty_output(self):
        self.assertEqual(rank_ocr([]).counts,{'empty_results':1})

    def test_format_rejected_counts(self):
        self.assertEqual(reading('NOISE').counts['format_rejected_results'],1)

    def test_tuple_compatibility(self):
        self.assertEqual(tuple(reading(score=.9)),('UP15AB1234',.9))
        self.assertEqual(coerce_reading((None,None),OCRConfig()).status,'unreadable')

    def test_duplicates_do_not_inflate_evidence(self):
        result=rank_ocr([(None,'UP15AB1234',.7),(None,'UP15AB1234',.9)])
        self.assertEqual(len(result.candidates),1)
        self.assertEqual(result.best.confidence,.9)

    def test_malformed_rows(self):
        self.assertEqual(rank_ocr([None,('x',),('b',None,.8)]).status,'format_unrecognized')


class ImageTests(unittest.TestCase):
    def setUp(self):
        self.image=np.arange(80*120*3,dtype=np.uint8).reshape(80,120,3)

    def test_valid_crop_is_copy(self):
        crop=safe_crop(self.image,(10,20,50,40))
        self.assertEqual(crop.shape,(20,40,3))
        self.assertFalse(np.shares_memory(crop,self.image))

    def test_partial_crop_clips(self):
        self.assertEqual(safe_crop(self.image,(-10,-20,50,40)).shape,(40,50,3))

    def test_zero_reversed_nonfinite_and_outside(self):
        for box in ((1,1,1,20),(20,1,1,20),(0,0,float('inf'),30),
                    (0,0,float('nan'),30),(0,0,'20',30),(200,200,300,300)):
            self.assertIsNone(safe_crop(self.image,box))

    def test_tiny_crop_and_configurable_limits(self):
        self.assertIsNone(safe_crop(self.image,(0,0,10,5)))
        self.assertEqual(safe_crop(self.image,(0,0,10,5),replace(OCRConfig(),min_width=5,min_height=3)).shape,(5,10,3))

    def test_empty_input(self):
        self.assertIsNone(safe_crop(np.empty((0,0,3),dtype=np.uint8),(0,0,30,20)))
        with self.assertRaises(ValueError):
            preprocess(np.empty((0,0,3),dtype=np.uint8))

    def test_all_methods_preserve_clean_original(self):
        original=self.image.copy()
        for method in ('color','gray','fixed','otsu','adaptive','contrast'):
            output=preprocess(self.image,method)
            self.assertTrue(output.size)
            self.assertFalse(np.shares_memory(output,self.image))
            np.testing.assert_array_equal(self.image,original)

    def test_fixed_threshold_matches_original(self):
        expected=cv2.threshold(cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY),64,255,cv2.THRESH_BINARY_INV)[1]
        np.testing.assert_array_equal(preprocess(self.image),expected)

    def test_otsu_and_adaptive_are_binary(self):
        for method in ('otsu','adaptive'):
            self.assertTrue(set(np.unique(preprocess(self.image,method))) <= {0,255})

    def test_controlled_upscale_and_validation(self):
        self.assertEqual(preprocess(self.image,'gray',2).shape,(160,240))
        for value in (0,4,float('nan'),True):
            with self.assertRaises(ValueError):
                preprocess(self.image,'gray',value)


class AssociationTests(unittest.TestCase):
    def test_single_vehicle(self):
        self.assertEqual(associate_plate((20,60,50,75),[(0,0,100,100,1)])[-1],1)

    def test_touching_boundary(self):
        self.assertEqual(associate_plate((0,60,30,75),[(0,0,100,100,1)])[-1],1)

    def test_partly_outside(self):
        self.assertEqual(associate_plate((-5,60,25,75),[(0,0,100,100,1)])[-1],1)

    def test_multiple_vehicles_and_plates(self):
        vehicles=[(0,0,100,100,1),(100,0,200,100,2)]
        self.assertEqual(associate_plate((20,60,50,75),vehicles)[-1],1)
        self.assertEqual(associate_plate((120,60,150,75),vehicles)[-1],2)

    def test_identical_overlap_is_ambiguous_order_independent(self):
        vehicles=[(0,0,100,100,1),(0,0,100,100,2)]
        self.assertEqual(associate_plate((20,60,50,75),vehicles)[-1],-1)
        self.assertEqual(associate_plate((20,60,50,75),vehicles[::-1])[-1],-1)

    def test_overlapping_but_clear_containment_wins(self):
        self.assertEqual(associate_plate((15,60,45,75),[(0,0,100,100,1),(35,0,135,100,2)])[-1],1)

    def test_nearby_uncontained_not_assigned(self):
        self.assertEqual(associate_plate((120,60,150,75),[(0,0,100,100,1)])[-1],-1)

    def test_duplicate_ids_invalid_geometry_and_plate_size(self):
        self.assertEqual(associate_plate((20,60,50,75),[(0,0,100,100,1)]*2)[-1],-1)
        self.assertEqual(associate_plate((0,0,100,100),[(0,0,100,100,1)])[-1],-1)
        self.assertEqual(associate_plate((1,1,1,3),[(0,0,100,100,1)])[-1],-1)

    def test_malformed_association_inputs_are_unassigned(self):
        self.assertEqual(associate_plate(None,[])[-1],-1)
        self.assertEqual(associate_plate((20,60,50,75),[None,'bad'])[-1],-1)


class ConsensusTests(unittest.TestCase):
    def setUp(self):
        self.config=replace(OCRConfig(),single_read_confidence=1)
        self.manager=OCRManager(self.config)
        self.event=event()

    def observe(self,frame=0,text='UP15AB1234',score=.8,plate=.9,target=None):
        target=target or self.event
        self.manager.due(target,frame,scan=True)
        self.manager.begin(target,frame,100)
        return self.manager.record(target,frame,reading(text,score,self.config),plate)

    def test_provisional_then_two_frame_consensus(self):
        self.assertIsNone(self.observe())
        self.assertEqual(self.event['recognition']['status'],'pending')
        self.assertEqual(self.observe(1),('UP15AB1234',.8))
        self.assertEqual(self.event['status'],'candidate')

    def test_equal_votes_are_conflicting(self):
        self.observe()
        self.assertIsNone(self.observe(1,'KA03MN1234'))
        self.assertEqual(self.event['recognition']['status'],'conflicting')

    def test_weighted_vote_requires_evidence_margin(self):
        self.observe(0,'UP15AB1234',.85)
        self.observe(1,'KA03MN1234',.7)
        self.assertEqual(self.observe(2,'UP15AB1234',.85),('UP15AB1234',.85))

    def test_duplicate_frame_not_consensus(self):
        self.observe()
        self.assertIsNone(self.manager.record(self.event,0,reading(),.9))
        self.assertEqual(len(self.manager.states['event-1'].history),1)

    def test_low_confidence_fragments_cannot_confirm(self):
        self.observe(0,'UP15',.99)
        self.observe(1,'AB1234',.99)
        self.observe(2,'UP15AB1234',.2)
        self.assertNotEqual(self.event['recognition']['status'],'accepted')

    def test_single_high_confidence_uncorrected_compatibility(self):
        manager=OCRManager()
        manager.due(self.event,0)
        manager.begin(self.event,0,1)
        self.assertEqual(manager.record(self.event,0,reading(score=.9),.9),('UP15AB1234',.9))
        self.assertFalse(manager.due(self.event,1))

    def test_correction_requires_repeat_even_high_confidence(self):
        self.assertIsNone(self.observe(0,'UPI5ABC1234',.99))
        self.assertEqual(self.observe(1,'UPI5ABC1234',.99),('UP15ABC1234',.99))

    def test_different_events_do_not_share_votes(self):
        self.observe()
        other=event(2)
        self.assertIsNone(self.observe(1,target=other))

    def test_history_size_and_expiry(self):
        self.manager=OCRManager(replace(self.config,min_observations=3,history_size=3,history_frames=2))
        for frame in range(7):
            self.observe(frame,'BAD')
        state=self.manager.states['event-1']
        self.assertEqual(len(state.history),3)
        self.assertEqual(state.history[0]['frame_index'],4)

    def test_state_ttl_and_capacity_are_bounded(self):
        self.manager=OCRManager(replace(self.config,max_states=1,state_ttl=2))
        self.observe()
        other=event(2)
        self.assertFalse(self.manager.due(other,1))
        self.manager.expire(3,{1:self.event})
        self.assertFalse(self.manager.states)
        self.assertTrue(self.event['recognition']['closed'])

    def test_interval_and_failed_attempt_backoff(self):
        self.observe(0,'BAD')
        self.assertTrue(self.manager.due(self.event,1))
        self.observe(1,'BAD')
        self.assertFalse(self.manager.due(self.event,2))
        self.assertTrue(self.manager.due(self.event,3))

    def test_better_crop_allows_earlier_retry(self):
        self.observe(0,'BAD')
        self.observe(1,'BAD')
        self.assertTrue(self.manager.due(self.event,2,quality=200))

    def test_maximum_attempts(self):
        self.manager=OCRManager(replace(self.config,max_attempts=1))
        self.observe(0,'BAD')
        self.assertFalse(self.manager.due(self.event,1))
        self.assertEqual(self.event['recognition']['reason'],'attempt_limit')

    def test_uncertain_identity_discards_evidence_not_event(self):
        self.observe()
        self.manager.invalidate(self.event)
        self.assertFalse(self.manager.due(self.event,1))
        self.assertEqual(self.event['event_id'],'event-1')
        self.assertEqual(self.event['status'],'candidate')

    def test_accepted_result_never_downgraded(self):
        self.observe()
        self.observe(1)
        before=self.event['recognition'].copy()
        self.assertIsNone(self.manager.record(self.event,2,reading(score=.7),.9))
        self.assertEqual(self.event['recognition'],before)

    def test_stronger_provisional_result_can_win(self):
        self.observe(0,score=.7)
        result=self.observe(1,score=.85)
        self.assertEqual(result,('UP15AB1234',.85))

    def test_two_vehicles_with_different_plates(self):
        other=event(2)
        self.observe(0)
        self.observe(0,'KA03MN1234',target=other)
        self.assertEqual(self.observe(1),('UP15AB1234',.8))
        self.assertEqual(self.observe(1,'KA03MN1234',target=other),('KA03MN1234',.8))

    def test_track_expiration_keeps_accepted_event(self):
        self.event.update(license_plate='UP15AB1234',license_plate_confidence=.95)
        self.manager.invalidate(self.event,'track_expired')
        self.assertEqual(self.event['license_plate'],'UP15AB1234')
        self.assertEqual(self.event['recognition']['status'],'accepted')


class ConfigurationTests(unittest.TestCase):
    def test_environment_configuration(self):
        cfg=OCRConfig.from_environment(environ={'TRAFFIC_OCR_CONFIDENCE':'.75','TRAFFIC_OCR_PREPROCESSING':'gray','TRAFFIC_OCR_ALLOW_LEGACY':'false'})
        self.assertEqual(cfg.ocr_confidence,.75)
        self.assertFalse(cfg.allow_legacy)
        self.assertEqual(cfg.preprocessing,'gray')

    def test_invalid_configuration(self):
        for values in ({'ocr_confidence':float('nan')},{'max_attempts':0},{'interval_frames':True},
                       {'min_observations':20},{'single_read_confidence':.3},{'preprocessing':'noise'},
                       {'upscale':float('inf')},{'max_corrections':4},{'allow_legacy':1}):
            with self.subTest(values=values),self.assertRaises(ValueError):
                OCRConfig(**values)

    def test_invalid_environment(self):
        with self.assertRaises(ValueError):
            OCRConfig.from_environment(environ={'TRAFFIC_OCR_DIAGNOSTIC':'yes'})

    def test_util_import_is_lightweight_and_offline(self):
        path=Path(__file__).resolve().parents[1]/'util.py'
        spec=importlib.util.spec_from_file_location('_phase5_util_test',path)
        module=importlib.util.module_from_spec(spec)
        fake=ModuleType('easyocr')
        fake.Reader=Mock(return_value=Mock())
        with patch.dict(sys.modules,{'easyocr':fake}):
            spec.loader.exec_module(module)
            fake.Reader.assert_not_called()
            module.get_reader()
        fake.Reader.assert_called_once_with(['en'],gpu=False,download_enabled=False)

    def test_raw_text_diagnostics_require_explicit_opt_in(self):
        import util
        fake=Mock()
        fake.readtext.return_value=[(None,'UP15AB1234',.9)]
        with patch.object(util,'reader',fake),patch.object(util,'logger') as logger:
            util.read_license_plate(np.zeros((20,40),dtype=np.uint8))
            logger.debug.assert_not_called()
            util.read_license_plate(np.zeros((20,40),dtype=np.uint8),replace(OCRConfig(),diagnostic=True))
            logger.debug.assert_called_once()


class ApplicationOCRTests(unittest.TestCase):
    def setUp(self):
        import test_stability as stability
        self.fixture=stability.ApplicationTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.app=stability.application
        self.result=stability.result
        self.addCleanup(patch.stopall)
        patch.object(self.app,'OCR_CONFIG',replace(OCRConfig(),single_read_confidence=1)).start()

    def prepare(self,count=2):
        capture=self.fixture.prepare_plate()
        capture.frames.extend([self.fixture.frame.copy() for _ in range(count-1)])
        capture.frame_count=count
        self.app.read_license_plate.return_value=reading()
        return capture

    def test_after_red_to_green_consensus_preserves_uuid_and_api(self):
        self.prepare()
        stream=self.app.generate_frames()
        next(stream)
        item=self.app.crossing_detector.events[7]
        identity=item['event_id']
        self.assertIsNone(item['license_plate'])
        self.app.app.test_client().post('/set_light/green')
        next(stream)
        self.assertEqual(item['event_id'],identity)
        self.assertEqual(item['status'],'candidate')
        self.assertEqual(self.app.app.test_client().get('/violations').get_json(),['UP15AB1234'])
        list(stream)

    def test_low_score_plate_does_not_call_ocr(self):
        self.prepare(1)
        self.app.license_plate_detector_model.return_value=self.result([[280,350,310,370,.1,0]])
        list(self.app.generate_frames())
        self.app.read_license_plate.assert_not_called()
        self.assertEqual(self.app.ocr_manager.counts['confidence_rejected_plates'],1)

    def test_failed_association_keeps_candidate(self):
        self.prepare(1)
        self.app.get_car.return_value=(-1,)*5
        list(self.app.generate_frames())
        self.assertEqual(len(self.app.crossing_detector.events),1)
        self.assertEqual(self.app.ocr_manager.counts['failed_associations'],1)

    def test_multiple_plate_boxes_only_one_attempt_per_frame(self):
        self.prepare(1)
        self.app.license_plate_detector_model.return_value=self.result([[280,350,310,370,.9,0],[280,350,315,372,.9,0]])
        list(self.app.generate_frames())
        self.app.read_license_plate.assert_called_once()
        self.assertEqual(self.app.crossing_detector.events[7]['recognition']['observations'],1)

    def test_small_crop_is_nonfatal_no_opencv_or_ocr(self):
        self.prepare(1)
        self.app.license_plate_detector_model.return_value=self.result([[280,350,285,353,.9,0]])
        self.assertEqual(len(list(self.app.generate_frames())),1)
        self.app.read_license_plate.assert_not_called()

    def test_failure_status_and_event_survive_exception(self):
        self.prepare(1)
        self.app.read_license_plate.side_effect=RuntimeError('engine failed')
        self.assertEqual(list(self.app.generate_frames()),[])
        self.assertEqual(self.app.crossing_detector.events[7]['recognition']['status'],'unreadable')
        self.assertEqual(self.app.crossing_detector.events[7]['status'],'candidate')

    def test_clean_inference_and_crop_survive_annotations(self):
        self.prepare(1)
        received=[]
        self.app.license_plate_detector_model.side_effect=lambda image:(received.append(image.copy()) or self.result([[280,350,310,370,.9,0]]))
        with patch.object(self.app.cv2,'line',side_effect=lambda image,*a,**k:image.fill(255)):
            list(self.app.generate_frames())
        self.assertFalse(received[0].any())
        np.testing.assert_array_equal(self.app.read_license_plate.call_args[0][0],np.full((20,30),255,dtype=np.uint8))

    def test_scheduling_limits_actual_calls(self):
        self.prepare(8)
        self.app.read_license_plate.return_value=rank_ocr([])
        list(self.app.generate_frames())
        self.assertEqual(self.app.read_license_plate.call_count,4) # frames 1,2,4,7

    def test_position_jump_closes_ocr_without_changing_candidate(self):
        self.prepare(2)
        stream=self.app.generate_frames()
        next(stream)
        item=self.app.crossing_detector.events[7]
        self.app.mot_tracker.update.return_value=np.asarray([[500,20,580,400,7]],dtype=float)
        next(stream)
        self.assertTrue(item['recognition']['closed'])
        self.assertIsNone(item['license_plate'])
        self.assertEqual(item['status'],'candidate')
        self.app.read_license_plate.assert_called_once()
        list(stream)

    def test_observation_gap_does_not_mix_identity_evidence(self):
        self.prepare(2)
        stream=self.app.generate_frames()
        next(stream)
        self.app._frame_index+=5
        next(stream)
        self.assertTrue(self.app.crossing_detector.events[7]['recognition']['closed'])
        self.app.read_license_plate.assert_called_once()
        list(stream)

    def test_invalid_startup_configuration_fails_before_capture(self):
        with patch.dict(os.environ,{'TRAFFIC_OCR_CONFIDENCE':'nan'}),patch.object(self.app.cv2,'VideoCapture') as capture:
            self.assertEqual(self.app.app.test_client().get('/video').status_code,503)
        capture.assert_not_called()

    def test_plate_threshold_is_applied_to_yolo_prefilter(self):
        capture=Mock()
        capture.isOpened.return_value=True
        vehicle,plate=Mock(),Mock()
        with patch.dict(os.environ,{'TRAFFIC_PLATE_CONFIDENCE':'.4'}), \
                patch.object(self.app.cv2,'VideoCapture',return_value=capture), \
                patch.object(self.app,'_load_yolo_model',side_effect=[vehicle,plate]), \
                patch.dict(sys.modules,self.fixture.dependencies()):
            self.app.initialize_resources()
        plate.overrides.update.assert_called_once_with(conf=.4)


if __name__=='__main__':
    unittest.main()
