"""Evaluate saved, explicitly labeled crop predictions; never loads OCR/Flask."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plate_recognition import normalize_text
from tools.detection_common import PROJECT_ROOT, sha256_file


def load_manifest(path, labels=False):
    path = Path(path).resolve()
    value = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(value, dict) or value.get('schema_version') != 1 or not isinstance(value.get('samples'), list) or not value['samples']:
        raise ValueError('Expected nonempty schema_version=1 samples list')
    samples = value['samples']
    ids = set()
    for sample in samples:
        if not isinstance(sample, dict) or not isinstance(sample.get('id'), str) or not sample['id'] or sample['id'] in ids:
            raise ValueError('Samples require unique nonempty string IDs')
        ids.add(sample['id'])
        if labels and (not isinstance(sample.get('text'), str) or not isinstance(sample.get('format'), str)):
            raise ValueError('Every labeled sample requires text (empty for negatives) and format')
        if labels and sample['text'] and not normalize_text(sample['text']):
            raise ValueError('Label must be normalizable ASCII text')
    return samples


def edit_distance(first, second):
    previous = list(range(len(second)+1))
    for index, char in enumerate(first, 1):
        current = [index]
        for offset, other in enumerate(second, 1):
            current.append(min(current[-1]+1, previous[offset]+1, previous[offset-1]+(char != other)))
        previous = current
    return previous[-1]


def metrics(labels, predictions):
    total, positive, accepted, wrong, exact, edits, characters = 0,0,0,0,0,0,0
    for sample, result in zip(labels, predictions):
        truth = normalize_text(sample['text'])
        text = normalize_text(result['text']) if result['accepted'] else ''
        total += 1
        positive += bool(truth)
        accepted += result['accepted']
        wrong += bool(result['accepted'] and (not truth or text != truth))
        exact += bool(truth and result['accepted'] and text == truth)
        if truth:
            edits += edit_distance(truth, text)
            characters += len(truth)
    return {'samples':total, 'positive_samples':positive, 'accepted':accepted,
            'exact_matches':exact, 'wrong_acceptances':wrong,
            'exact_match_accuracy':exact/positive if positive else None,
            'character_error_rate':edits/characters if characters else None,
            'acceptance_rate':accepted/total, 'rejection_rate':1-accepted/total,
            'false_acceptance_rate':wrong/accepted if accepted else None,
            'false_acceptance_per_sample':wrong/total}


def evaluate(labels, predictions):
    by_id = {sample['id']:sample for sample in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {sample['id'] for sample in labels}:
        raise ValueError('Predictions must cover exactly the labeled IDs')
    strategies = None
    for sample in predictions:
        results = sample.get('strategies')
        if not isinstance(results, dict) or not results:
            raise ValueError('Every prediction needs named strategies')
        if strategies is None:
            strategies = set(results)
        if set(results) != strategies:
            raise ValueError('Every strategy must cover identical samples')
        for result in results.values():
            if (not isinstance(result, dict) or not isinstance(result.get('accepted'), bool)
                    or not isinstance(result.get('text'), str) or (result['accepted'] and not normalize_text(result['text']))):
                raise ValueError('Predictions need text and boolean accepted; accepted text cannot be empty')
    rows = []
    for strategy in sorted(strategies):
        ordered = [by_id[sample['id']]['strategies'][strategy] for sample in labels]
        rows.append(dict(strategy=strategy, group='all', **metrics(labels, ordered)))
        for kind in sorted({sample['format'] for sample in labels}):
            indices = [i for i,sample in enumerate(labels) if sample['format']==kind]
            rows.append(dict(strategy=strategy, group=kind,
                             **metrics([labels[i] for i in indices],[ordered[i] for i in indices])))
    stages = {}
    for field in ('plate_detected', 'correct_association'):
        values = [sample.get(field) for sample in predictions]
        if any(value is not None and not isinstance(value, bool) for value in values):
            raise ValueError('Stage observations must be booleans')
        # Report only explicit fully observed positive samples; no inferred truth.
        selected = [by_id[s['id']].get(field) for s in labels if s['text']]
        stages[field] = sum(selected)/len(selected) if selected and all(isinstance(x,bool) for x in selected) else None
    return {'metrics':rows, 'stage_observation_rates':stages,
            'scope':'Labeled isolated crops. Stage rates require explicit independently labeled booleans; '
                    'crop OCR accuracy is not detector recall or vehicle association accuracy. '
                    'End-to-end exact-plate accuracy is unmeasured without matched video event truth.'}


def write_report(path, report, protected=()):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT/path
    path = path.resolve()
    if PROJECT_ROOT/'outputs' not in path.parents:
        raise ValueError('Generated reports must be inside ignored outputs/')
    if path.exists() or path in {Path(p).resolve() for p in protected}:
        raise ValueError('Refusing to overwrite existing file/input')
    if path.suffix.lower() not in ('.json','.csv'):
        raise ValueError('Report extension must be .json or .csv')
    if path.suffix.lower()=='.csv':
        if not report.get('metrics'):
            raise ValueError('CSV export requires evaluation metrics')
        rows = [dict(row, annotations_sha256=report.get('annotations_sha256',''),
                     predictions_sha256=report.get('predictions_sha256',''),
                     scope=report.get('scope','')) for row in report['metrics']]
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        payload = buffer.getvalue()
    else:
        payload = json.dumps(report,indent=2,allow_nan=False)+'\n'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8',newline='') as handle:
        handle.write(payload)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations',required=True)
    parser.add_argument('--predictions',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:
        report=evaluate(load_manifest(args.annotations,True),load_manifest(args.predictions))
        report.update(annotations_sha256=sha256_file(args.annotations),predictions_sha256=sha256_file(args.predictions))
        write_report(args.output,report,(args.annotations,args.predictions))
        print('OCR evaluation written; '+str(len(report['metrics']))+' metric groups')
        return 0
    except (OSError,ValueError,TypeError,KeyError) as error:
        parser.exit(1,'OCR evaluation failed: '+str(error)+'\n')


if __name__=='__main__':
    raise SystemExit(main())
