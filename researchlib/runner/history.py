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
