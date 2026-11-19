"""Offline cached-EasyOCR crop replay, identical samples per preprocessing strategy."""
import argparse
from dataclasses import asdict, replace
from pathlib import Path
import platform
import sys
import time

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from plate_recognition import METHODS, OCRConfig, preprocess, rank_ocr
from tools.benchmark_detection import latency_summary
from tools.detection_common import offline_environment, sha256_file
from tools.evaluate_ocr import evaluate, load_manifest, write_report


def load_images(samples, manifest):
    images=[]
    for sample in samples:
        if not isinstance(sample.get('image'),str):
            raise ValueError('Each benchmark sample needs an image path')
        path=(Path(manifest).resolve().parent/sample['image']).resolve()
        if not path.is_file():
            raise ValueError('Missing local plate image: '+str(path))
        image=cv2.imread(str(path))
        if image is None or not image.size:
            raise ValueError('Unreadable local plate image: '+str(path))
        images.append((image,sha256_file(path)))
    return images


def benchmark(reader, samples, images, methods=METHODS, repeats=1, warmup=1, config=None,
              processed_frames=None, violation_candidates=None):
    config=config or OCRConfig()
    if not samples or len(samples)!=len(images) or len({s['id'] for s in samples})!=len(samples):
        raise ValueError('Unique samples and corresponding images required')
    if not methods or len(set(methods))!=len(methods) or any(m not in METHODS for m in methods):
        raise ValueError('Select distinct supported preprocessing methods')
    if isinstance(repeats,bool) or not isinstance(repeats,int) or repeats<1 or isinstance(warmup,bool) or not isinstance(warmup,int) or warmup<0:
        raise ValueError('Invalid repeat/warmup count')
    for count in (processed_frames,violation_candidates):
        if count is not None and (isinstance(count,bool) or not isinstance(count,int) or count<1):
            raise ValueError('Optional denominators must be positive integers')
    predictions=[{'id':s['id'],'strategies':{}} for s in samples]
    runs=[]
    for method in methods:
        for i in range(warmup):
            reader.readtext(preprocess(images[i%len(images)][0],method,config.upscale))
        timings, overhead=[],[]
        counts={}
        start_all=time.perf_counter()
        for repeat in range(repeats):
            for i,(image,digest) in enumerate(images):
                start=time.perf_counter()
                prepared=preprocess(image,method,config.upscale)
                overhead.append(time.perf_counter()-start)
                start=time.perf_counter()
                output=reader.readtext(prepared)
                timings.append(time.perf_counter()-start)
                reading=rank_ocr(output,config)
                for name,count in reading.counts.items():
                    counts[name]=counts.get(name,0)+count
                if repeat==0:
                    best=reading.best
                    predictions[i]['strategies'][method]={'text':best.normalized_text if best else '',
                        'accepted':best is not None,'status':reading.status,
                        'confidence':best.confidence if best else None}
        elapsed=time.perf_counter()-start_all
        calls=len(samples)*repeats
        runs.append({'strategy':method,'upscale':config.upscale,'calls':calls,'warmup_calls':warmup,
                     'ocr_latency':latency_summary(timings),'preprocessing_latency':latency_summary(overhead),
                     'crop_pipeline_fps':calls/elapsed,'elapsed_seconds':elapsed,
                     'ocr_calls_per_frame':calls/(processed_frames*repeats) if processed_frames else None,
                     'ocr_calls_per_candidate':calls/(violation_candidates*repeats) if violation_candidates else None,
                     'counts':counts})
    report={'schema_version':1,'samples':predictions,'runs':runs,'configuration':asdict(config),
            'sample_sha256':{s['id']:image[1] for s,image in zip(samples,images)},
            'hardware':{'platform':platform.platform(),'cpu':platform.processor(),'python':platform.python_version(),
                        'opencv':cv2.__version__,'numpy':np.__version__,'device':str(getattr(reader,'device','unknown'))},
            'scope':'Isolated clean plate crops; all methods use identical images. Initialization/decoding/warmup excluded. '
                    'Includes preprocessing, OCR, ranking; excludes detector/tracker/Flask. '
                    'Acceptance is per-crop format/confidence eligibility, not production multi-frame event consensus. '
                    'Predictions use first measured repeat; timings/counts use all measured repeats.'}
    if all(isinstance(s.get('text'),str) and isinstance(s.get('format'),str) for s in samples):
        report.update(evaluate(samples,predictions))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=['color','gray','fixed','otsu','adaptive','contrast'])
    parser.add_argument('--upscale',type=float,default=1)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--warmup',type=int,default=1)
    parser.add_argument('--processed-frames',type=int)
    parser.add_argument('--violation-candidates',type=int)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        if Path(args.output).suffix.lower()!='.json':
            raise ValueError('Benchmark output must be JSON')
        samples=load_manifest(args.manifest)
        images=load_images(samples,args.manifest)
        config=replace(OCRConfig(),upscale=args.upscale)
        if args.repeats < 1 or args.warmup < 0 or len(set(args.methods)) != len(args.methods):
            raise ValueError('Repeats must be positive, warmup nonnegative, methods distinct')
        if any(value is not None and value < 1 for value in (args.processed_frames,args.violation_candidates)):
            raise ValueError('Optional denominators must be positive')
        with offline_environment():
            from util import get_reader
            report=benchmark(get_reader(),samples,images,args.methods,args.repeats,args.warmup,config,
                             args.processed_frames,args.violation_candidates)
        report['manifest_sha256']=sha256_file(args.manifest)
        write_report(args.output,report,(args.manifest,))
        print('OCR benchmark written for '+str(len(samples))+' identical samples')
        return 0
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1,'OCR benchmark failed: '+str(error)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
