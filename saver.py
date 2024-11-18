import openpyxl
import os
import sys
from torch.utils.tensorboard import SummaryWriter


class ValueSaver():
    '''
    Save value results to excel and summarywriter
    '''
    def __init__(self, dir_result, kind_list, col_list):
        '''
        for example, 
        kind = ['train','val', ...]
        col = ['loss', ...] 
        '''
        self.kind_list = kind_list
        self.col_list = col_list

        self.epoch = None
        self.result = None
        
        # Init result excel file
        self.col_result = ['epoch']
        for k in kind_list:
            self.col_result.extend([k + '__' + x for x in col_list])

        self.path_excel = os.path.join(dir_result, 'results.xlsx') 
        if os.path.isfile(self.path_excel) is False:
            wb = openpyxl.Workbook()
            worksheet = wb.active
            worksheet.append(self.col_result)
            wb.save(self.path_excel)

        # Init summarywriter
        dir_tensorboard = os.path.join(dir_result, 'tensorboard')
        if os.path.isdir(dir_tensorboard) is False: os.mkdir(dir_tensorboard)

        self.summary_writer_dict = {}
        for kind in self.kind_list:
            dir_tensorboard_kind = os.path.join(dir_tensorboard, kind)
            if os.path.isdir(dir_tensorboard_kind) is False: os.mkdir(dir_tensorboard_kind)

            self.summary_writer_dict[kind] = SummaryWriter(dir_tensorboard_kind)

    def ready(self, epoch):
        self.epoch = epoch
        self.init_result()
    
    def init_result(self):
        self.result = {}
        for k in self.kind_list:
            self.result[k] = {}
            for c in self.col_list:
                self.result[k][c] = 0.0
        
    def update_result(self, kind, col, val, idx_step=None, flag_step=False):
        self.result[kind][col] += val

        if idx_step is not None and flag_step:
            self.save_per_step(idx_step, kind, col, val)

    def save_per_step(self, idx_step, kind, col, val):
        summary_writer = self.summary_writer_dict[kind]
        summary_writer.add_scalar('step/'+str(col), val, idx_step)

    def save_per_epoch(self):
        # Save results to excel
        wb = openpyxl.load_workbook(self.path_excel)
        ws = wb.active
        results_list = []
        for col_excel in self.col_result:
            if col_excel == 'epoch': results_list.append(self.epoch)
            else:
                kind = col_excel[:col_excel.find('__')]
                col = col_excel[col_excel.find('__')+2:]
                # print(kind)
                # print(col)
                results_list.append(self.result[kind][col]) 
        ws.append(results_list)
        wb.save(self.path_excel)

        # Save results to tensorboard
        for kind in self.kind_list:
            summary_writer = self.summary_writer_dict[kind]
            for col in self.col_list:
                if self.result[kind][col] != 'None': 
                    summary_writer.add_scalar('epoch/'+col, self.result[kind][col], self.epoch) 

    def get_epoch_val(self, kind, col):
        if self.result[kind][col] is None:
            print('result is None.')
            sys.exit()
        else:
            return self.result[kind][col]

