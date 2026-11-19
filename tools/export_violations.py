"""Local filtered SQLite export; registration numbers/evidence are opt-in."""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
from uuid import uuid4

if __package__ in (None,''):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from database import Database, StorageError


ROOT = Path(__file__).resolve().parents[1]
BASE_FIELDS = ('event_id','session_id','detected_at','frame_index','confirmation_frame_index',
               'video_timestamp_ms','timestamp_kind','vehicle_category','signal_state',
               'crossing_point','detection_status','review_status','recognition_status','purge_pending')


def csv_cell(value):
    if value is None:
        return ''
    if isinstance(value,(dict,list,tuple)):
        value = json.dumps(value,separators=(',',':'),ensure_ascii=True,allow_nan=False)
    if isinstance(value,str):
        stripped = value
        while stripped and (stripped[0].isspace() or ord(stripped[0])<32 or stripped[0]=='\ufeff'):
            stripped = stripped[1:]
        if (value and (ord(value[0])<32 or value[0]=='\ufeff')) or (stripped and stripped[0] in '=+-@'):
            return "'"+value
    return value


def export_records(repository,destination,filters,limit=100,offset=0,
                   include_plates=False,include_evidence=False,include_review_notes=False):
    if not filters:
        raise ValueError('Select at least one history filter for export')
    if not all(isinstance(value,bool) for value in (include_plates,include_evidence,include_review_notes)):
        raise ValueError('Sensitive export flags require booleans')
    destination = Path(destination)
    destination = (destination if destination.is_absolute() else ROOT/destination).resolve()
    if ROOT/'outputs' not in destination.parents or destination.suffix.lower() not in ('.json','.csv'):
        raise ValueError('Export destination must be .csv/.json inside outputs/')
    if destination.exists() or destination==repository.path:
        raise ValueError('Refusing to overwrite an existing file or database')
    records = repository.history(limit=limit,offset=offset,**filters)
    fields = list(BASE_FIELDS)
    if include_plates:
        fields += ['license_plate','automatic_plate','initial_accepted_plate','ocr_confidence','format_type']
    if include_evidence:
        fields.append('evidence')
    if include_review_notes:
        fields += ['review_note','reviewed_at','review_history']
    rows = [{key:record.get(key) for key in fields} for record in records]
    if destination.suffix.lower()=='.csv':
        buffer = io.StringIO(newline='')
        writer = csv.DictWriter(buffer,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:csv_cell(row[key]) for key in fields} for row in rows)
        payload = buffer.getvalue()
    else:
        payload = json.dumps({'schema_version':1,'count':len(rows),'limit':limit,'offset':offset,
                              'sensitive_fields_included':{'plates':include_plates,'evidence':include_evidence,
                                                           'review_notes':include_review_notes},
                              'records':rows},indent=2,allow_nan=False)+'\n'
    destination.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    created = False
    try:
        with destination.open('x',encoding='utf-8',newline='') as handle:
            created = True
            handle.write(payload)
    except OSError:
        if created:
            destination.unlink()
        raise StorageError('export_unavailable') from None
    return len(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--database',default=str(ROOT/'data'/'violations.db'))
    parser.add_argument('--output',default=None)
    parser.add_argument('--format',choices=('csv','json'),default='json')
    parser.add_argument('--session-id'); parser.add_argument('--event-id')
    parser.add_argument('--after'); parser.add_argument('--before')
    parser.add_argument('--review-status'); parser.add_argument('--recognition-status')
    parser.add_argument('--license-plate'); parser.add_argument('--vehicle-category'); parser.add_argument('--signal-state')
    parser.add_argument('--limit',type=int,default=100); parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--include-plates',action='store_true')
    parser.add_argument('--include-evidence',action='store_true')
    parser.add_argument('--include-review-notes',action='store_true')
    args = parser.parse_args(argv)
    try:
        names = ('session_id','event_id','after','before','review_status','recognition_status',
                 'license_plate','vehicle_category','signal_state')
        filters = {name:getattr(args,name) for name in names if getattr(args,name) is not None}
        destination = args.output or 'outputs/violations-'+uuid4().hex+'.'+args.format
        count = export_records(Database(args.database),destination,filters,args.limit,args.offset,
                               args.include_plates,args.include_evidence,args.include_review_notes)
        print('Exported '+str(count)+' selected violation records; no external upload')
        return 0
    except (StorageError,ValueError,OSError) as error:
        message = error.code if isinstance(error,StorageError) else 'invalid_arguments_or_destination'
        parser.exit(1,'Violation export failed ('+message+')\n')


if __name__=='__main__':
    raise SystemExit(main())
