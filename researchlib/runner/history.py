class History:
    def __init__(self, records=None):
        self.records = {}
        if records:
            self.records = records
    
    def __add__(self, target):
        for key in target.records:
            self.records.setdefault(key, [])
            self.records[key] += target.records[key]
        return History(self.records)
    
    def add(self, d, prefix=None):
        for key in d:
            if prefix:
                ckey = prefix + '_' + key
            else:
                ckey = key
            self.records.setdefault(ckey, [])
            self.records[ckey].append(d[key])
            