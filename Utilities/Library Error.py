class LibraryError(Exception):
    def __init__(self, keys, stat_names):
        self.msg = 'Stored Distributions: ' + keys + '\nSelected Distributions: ' + stat_names

    def __str__(self):
        return self.msg
