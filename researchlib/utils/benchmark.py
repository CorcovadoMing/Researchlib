import os
import json
import datetime as dt
from pytz import timezone
import pygsheets

from .class_lib import _register_method
__methods__ = []
register_method = _register_method(__methods__)

module_path = os.path.dirname(os.path.abspath(__file__))
client_secret_path = os.path.join(
    module_path,
    '../.credential/client_secret_855251407115-slk3rv4qfi6b6narp2hapov9lv6iipjt.apps.googleusercontent.com.json'
)
sheetnames2ids_path = os.path.join(module_path,
                                   '../.credential/sheetnames2ids.json')


class Singleton:

    def __init__(self, klass):
        self.klass = klass
        self.instance = None

    def __call__(self, *args, **kwds):
        if self.instance == None:
            self.instance = self.klass(*args, **kwds)
        return self.instance


@Singleton
class benchmark(object):
    """ Document the best practice on Google sheet
    
    Attrs:
        self.gc (object): google api controller
        self.sheetnames2ids (dict): a mappting table mapping sheets' names to ids.
            (safer to use id than name because google drive allow duplicate names)
        self._team_id (str): (private), team folder id, please don't change
        self._folder_id (str): (private), team folder id, please don't change
        self._client_secret_path (str): (private), path to Google API credentials, please don't change
    """

    def __init__(self,
                 client_secret_path=client_secret_path,
                 sheetnames2ids_path=sheetnames2ids_path,
                 daily_backup=False):
        ## 1. benchmark settings ##
        self._client_secret_path = client_secret_path
        self.sheetnames2ids_path = sheetnames2ids_path
        self._team_id = '0ALUbsGKbWAJaUk9PVA'
        self._folder_id = '1ISC98c-oCxi-O-6cIOnQNSEhdcSXWk9K'
        self._backup_folder_id = '1nzOMdps9s-nD_YNKCcCwW79mhIdg_AM2'
        self._back_up_postfix = '_backup'
        self._worksheet_name = 'default'
        self._primary_keys = [
            'time_id',
        ]
        self._date_format = '%Y/%m/%d %H:%M:%S'
        self._date_timezone = 'Asia/Taipei'
        # benchmark sheets
        self.sheetnames = [
            'Classification', 'Segmentation', 'GAN', 'AnomalyDetection'
        ]
        # for auth
        self.gc = None

        ## 2. sheets settings ##
        try:
            with open(self.sheetnames2ids_path, 'r') as f:
                self.sheetnames2ids = json.load(f)
        except:
            print('creating sheetnames2ids.json(first time only), save at:\n{}'
                  .format(self.sheetnames2ids_path))
            self.sheetnames2ids = {}
            for sheetname in self.sheetnames:
                self.sheetnames2ids[sheetname] = ''
            self.genesis(self.sheetnames2ids.keys())
            with open(self.sheetnames2ids_path, 'w') as f:
                json.dump(self.sheetnames2ids, f)

        ## 3. others ##
        if daily_backup:
            try:
                self.daily_backup()
            except:
                print('Daily backup failed!!')

    def get_date(self):
        return dt.datetime.now(timezone(self._date_timezone)).strftime(
            self._date_format)

    def get_auth(self):
        try:
            self.gc = pygsheets.authorize(
                client_secret=self._client_secret_path)
            self.gc.drive.enable_team_drive(self._team_id)  # enable TeamDrive
            print('Google API Auth Successfully!')
        except:
            print('authendication failed, no internet connection?')


    def daily_backup(self):
        for sheetname in self.sheetnames2ids:
            self.verify(sheetname)
            try:
                sh = self.gc.open(sheetname + self._back_up_postfix)
                wks = sh.worksheet_by_title(self._worksheet_name)
                time_id_list = wks.get_col(
                    col=1, returnas='matrix', include_tailing_empty=False)
                if len(time_id_list) > 0:
                    time_id_list = time_id_list[1:]  # exclude header
                    last_modified_date = dt.datetime.strptime(
                        time_id_list[0], self._date_format)
                    if dt.datetime.now(timezone(self._date_timezone)).date(
                    ) > last_modified_date.date():
                        self.backup(sheetname)
                    else:
                        print('{} benchmark is backed up to today.'.format(
                            sheetname))
            except:
                self.backup(sheetname)

    def backup(self, sheetname):
        self.verify(sheetname)
        print('backing up {} benchmark'.format(sheetname))
        self.gc.drive.copy_file(
            file_id=self.sheetnames2ids[sheetname],
            title=sheetname + self._back_up_postfix,
            folder=self._backup_folder_id)

    def verify(self, sheetname):
        """ make sure the auth exists and sheetname following the rule!
        """
        if self.gc is None:
            self.get_auth()
        if sheetname not in self.sheetnames2ids:
            raise ValueError('{} is not in the {}'.format(
                sheetname, self.sheetnames2ids.keys()))

    def genesis(self, sheetnames):
        """ delete sheet if exist, then create new one.  
        Args:
            sheetnames (list): spreadsheets' name
        """
        for sheetname in sheetnames:
            self.verify(sheetname)

        ans = input(
            'Are you sure reset these benchmarks: {}? [y/n]'.format(sheetnames))
        if ans in ['y', 'Y']:
            for sheetname in sheetnames:
                # delete if not exist
                try:
                    sh = self.gc.open(sheetname)
                    sh.delete()
                    print('Delete {}'.format(sheetname))
                except:
                    print(
                        'Failed to delete {}, because it is not exist.'.format(
                            sheetname))
                # create new
                sh = self.gc.create(sheetname, folder=self._folder_id)
                self.sheetnames2ids[sheetname] = sh.id
                wks = sh.worksheet('index', 0)
                wks.title = self._worksheet_name
                wks.rows = 2
                wks.cols = 2
                #                 wks = sh.add_worksheet(self._worksheet_name, rows=1, cols=2)
                wks.frozen_rows = 1
                wks.frozen_cols = 1
                print('Create {}'.format(sheetname))
        else:
            print('Okay, no one get hurt.')

    def update_from_runner(self, sheetname, time_id, description, backup=False):
        """ update row(description) by primary key(time_id) in spreadsheet(sheetname)
        Args:
            time_id (str): primary key, use the creation timestamp of the runner.
            sheetname (str): sheetname of the google spreadsheet
            description (dict): config of runner
        """
        self.verify(sheetname)
        
        sh = self.gc.open_by_key(self.sheetnames2ids[sheetname])
        wks = sh.worksheet_by_title(self._worksheet_name)
        worksheet_cols = wks.get_row(
            row=1, returnas='matrix', include_tailing_empty=False)
        if len(worksheet_cols) == 0:
            worksheet_cols = self._primary_keys + worksheet_cols
        new_cols = list(description.keys())
        # filter out new cols
        diff_cols = list(set(new_cols) - set(worksheet_cols))
        if len(diff_cols) > 0:
            # update worksheet_cols
            worksheet_cols = worksheet_cols + diff_cols
            wks.update_row(index=1, values=worksheet_cols, col_offset=0)

        # recollect the values by keys' order
        insert_values = [
            time_id,
        ]  # for self._primary_keys
        assert len(insert_values) == len(self._primary_keys)
        for col in worksheet_cols[len(self._primary_keys):]:
            if col in description:
                insert_values.append(str(description[col]))
            else:
                insert_values.append('')

        # if time_id exists, update the row by time_id
        time_id_list = wks.get_col(
            col=1, returnas='matrix', include_tailing_empty=False)
        try:
            time_id_idx = time_id_list.index(time_id)
        except:
            time_id_idx = -1

        if time_id_idx >= 0:
            index_ = time_id_idx + wks.frozen_rows
            wks.update_row(index=index_, values=insert_values, col_offset=0)
        else:
            # always insert at the top of worksheet
            wks.insert_rows(
                wks.frozen_rows, number=1, values=insert_values, inherit=True)

        if backup:
            self.backup(sheetname)
