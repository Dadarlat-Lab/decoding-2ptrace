import sys

'''
path_log = os.path.join(dir_result, 'log.txt')
sys.stdout = Logger(path_log)
print('anything')
'''

class Logger:
    def __init__(self, filename):
        self.console = sys.stdout
        self.file = open(filename, 'w')
 
    def write(self, message):
        self.console.write(message)
        self.file.write(message)
 
    def flush(self):
        self.console.flush()
        self.file.flush()
    
    def close_logger(self):
        sys.stdout = self.console
        self.file.close()
