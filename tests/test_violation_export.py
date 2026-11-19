"""Selected-record exports with explicit sensitive-field flags and safe CSV cells."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from database import Database
from tools.export_violations import ROOT, csv_cell, export_records, main
from test_database import candidate, session


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.scratch=tempfile.TemporaryDirectory(dir=ROOT/'outputs',prefix='violation-export-test-')
        self.addCleanup(self.scratch.cleanup)
        self.root=Path(self.scratch.name)
        self.db=Database(self.root/'test.db');self.db.initialize()
        self.session=session();self.db.create_session(self.session)
        self.record=candidate(self.session['session_id'])
        self.db.save_event(self.record,{'status':'accepted','license_plate':'UP15AB1234','confidence':.95})
        self.filters={'session_id':self.session['session_id']}

    def test_default_json_omits_sensitive_fields(self):
        path=self.root/'export.json'
        export_records(self.db,path,self.filters)
        value=json.loads(path.read_text())
        self.assertEqual(value['count'],1)
        self.assertNotIn('UP15AB1234',path.read_text())
        self.assertNotIn('license_plate',value['records'][0])
        self.assertNotIn('evidence',value['records'][0])

    def test_explicit_plate_export_and_missing_values(self):
        path=self.root/'export.json'
        export_records(self.db,path,self.filters,include_plates=True)
        row=json.loads(path.read_text())['records'][0]
        self.assertEqual(row['license_plate'],'UP15AB1234')
        self.assertIsNone(row['vehicle_category'])

    def test_csv_columns_and_formula_escaping(self):
        self.db.review_event(self.record['event_id'],'pending','operator',note='=HYPERLINK("unsafe")')
        path=self.root/'export.csv'
        export_records(self.db,path,self.filters,include_review_notes=True)
        with path.open(newline='') as handle:
            row=next(csv.DictReader(handle))
        self.assertTrue(row['review_note'].startswith("'="))
        self.assertNotIn('license_plate',row)

    def test_all_dangerous_csv_prefixes(self):
        for value in ('=SUM(1,2)','+1','-1','@name','\t=1','\n=1','  =1','\ufeff=1','\x00=1','\v=1','\f=1','\u00a0=1'):
            self.assertTrue(csv_cell(value).startswith("'"))
        self.assertEqual(csv_cell('UP15AB1234'),'UP15AB1234')
        self.assertEqual(csv_cell(None),'')

    def test_empty_json_and_csv(self):
        for extension in ('json','csv'):
            path=self.root/('empty.'+extension)
            self.assertEqual(export_records(self.db,path,{'review_status':'approved'}),0)
            self.assertTrue(path.stat().st_size)

    def test_export_requires_filter(self):
        with self.assertRaises(ValueError):
            export_records(self.db,self.root/'export.json',{})

    def test_existing_destination_never_overwritten(self):
        path=self.root/'existing.json';path.write_text('existing')
        with self.assertRaises(ValueError):
            export_records(self.db,path,self.filters)
        self.assertEqual(path.read_text(),'existing')

    def test_outside_outputs_and_traversal_rejected(self):
        for path in (ROOT/'data'/'export.json',ROOT/'outputs'/'..'/'export.csv',self.root/'bad.txt'):
            with self.assertRaises(ValueError):
                export_records(self.db,path,self.filters)

    def test_pagination_exports_selected_page(self):
        self.assertEqual(export_records(self.db,self.root/'export.json',self.filters,limit=1,offset=1),0)

    def test_cli_does_not_upload_or_initialize_models(self):
        path=self.root/'cli.csv'
        self.assertEqual(main(['--database',str(self.db.path),'--output',str(path),
                              '--session-id',self.session['session_id']]),0)
        self.assertTrue(path.exists())

    def test_export_database_read_does_not_create_missing_db(self):
        missing=Database(self.root/'missing.db')
        from database import StorageError
        with self.assertRaises(StorageError):
            export_records(missing,self.root/'export.json',self.filters)
        self.assertFalse(missing.path.exists())


if __name__=='__main__':
    unittest.main()
