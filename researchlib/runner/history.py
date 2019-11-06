import os


class History:
    def __init__(self, records = None):
        self.records = {}
        if records:
            self.records = records

    def __add__(self, target):
        for key in target.records:
            self.records.setdefault(key, [])
            try:
                self.records[key] += list(map(float, target.records[key]))
            except:
                self.records[key] += list(map(str, target.records[key]))
        return History(self.records)

    def add(self, d, prefix = None):
        for key in d:
            if prefix:
                ckey = prefix + '_' + key
            else:
                ckey = key
            self.records.setdefault(ckey, [])
            try:
                self.records[ckey].append(float(d[key]))
            except:
                self.records[ckey].append(str(d[key]))
    
    def start_logfile(self, file_path):
        self.logfilename = os.path.join(file_path, 'log.csv')
    
    def update_logfile(self):
        recent = list(self.records.items())
        epoch = len(recent[0][1])
        if epoch == 1:
            with open(self.logfilename, 'a') as f:
                f.write(','.join(['epoch'] + list(self.records.keys())) + '\n')
        recent = [str(epoch)] + [str(j[-1]) for i, j in recent]
        recent = ','.join(recent)
        with open(self.logfilename, 'a') as f:
            f.write(recent + '\n')
        
