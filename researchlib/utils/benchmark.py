import os
import pygsheets

from .class_lib import _register_method
__methods__ = []
register_method = _register_method(__methods__)

module_path = os.path.dirname(os.path.abspath(__file__))
client_secret_path = os.path.join(
    module_path,
    '../.credential/client_secret_855251407115-slk3rv4qfi6b6narp2hapov9lv6iipjt.apps.googleusercontent.com.json'
)


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
        self.categories (list): list of all benchmarks
        self._team_id (str): (private), team folder id, please don't change
        self._folder_id (str): (private), team folder id, please don't change
        self._client_secret_path (str): (private), path to Google API credentials, please don't change
    """

    def __init__(self, client_secret_path=client_secret_path):
        self._client_secret_path = client_secret_path
        self.categories = [
            'Classification', 'Segmentation', 'GAN', 'AnomalyDetection'
        ]
        self.gc = pygsheets.authorize(client_secret=self._client_secret_path)
        # open enable TeamDrive support
        self._team_id = '0ALUbsGKbWAJaUk9PVA'
        self._folder_id = '1ISC98c-oCxi-O-6cIOnQNSEhdcSXWk9K'
        self.gc.drive.enable_team_drive(self._team_id)
        self._worksheet_name = 'default'
        self._primary_keys = [
            'time_id',
        ]

    def verify_name(self, name):
        if name not in self.categories:
            raise ValueError('{} is not in the {}'.format(
                name, self.categories))

    def genesis(self, categories):
        """ delete sheet if exist, then create new one.  
        Args:
            categories (list): spreadsheets' name
        """
        for cat in categories:
            self.verify_name(cat)

        ans = input(
            'Are you sure reset these benchmarks: {}? [y/n]'.format(categories))
        if ans in ['y', 'Y']:
            for cat in categories:
                # delete if not exist
                try:
                    sh = self.gc.open(cat)
                    sh.delete()
                    print('Delete {}'.format(cat))
                except:
                    print(
                        'Failed to delete {}, because it is not exist.'.format(
                            cat))
                # create new
                sh = self.gc.create(cat, folder=self._folder_id)
                wks = sh.worksheet('index', 0)
                wks.title = self._worksheet_name
                wks.rows = 2
                wks.cols = 2
                #                 wks = sh.add_worksheet(self._worksheet_name, rows=1, cols=2)
                wks.frozen_rows = 1
                wks.frozen_cols = 1
                print('Create {}'.format(cat))
        else:
            print('Okay, no one get hurt.')

    def update_from_runner(self, category, time_id, description):
        """ update row(description) by primary key(time_id) in spreadsheet(category)
        Args:
            time_id (str): primary key, use the creation timestamp of the runner.
            category (str): name of the google spreadsheet
            description (dict): config of runner
        """
        sh = self.gc.open(category)
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
