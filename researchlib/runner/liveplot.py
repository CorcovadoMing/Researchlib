from ipywidgets import IntProgress, Output, HBox, Label
from IPython import display as _display
import hiddenlayer as hl
import threading
from pynvml import *
import time
from texttable import Texttable
from ..utils import Timer
import redis
import pickle
from tqdm.auto import tqdm
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def _get_gpu_monitor(index):
    handle = nvmlDeviceGetHandleByIndex(index)
    info = nvmlDeviceGetMemoryInfo(handle)
    s = nvmlDeviceGetUtilizationRates(handle)
    return int(100 * (info.used / info.total)), s.gpu


def _gpu_monitor_worker(membar, utilsbar):
    while True:
        global _STOP_GPU_MONITOR_
        if _STOP_GPU_MONITOR_:
            for i, (membar_i, utilsbar_i) in enumerate(zip(membar, utilsbar)):
                membar_i.value, utilsbar_i.value = 0, 0
                membar_i.description, utilsbar_i.description = 'M' + str(i), 'U' + str(i)
            break
        else:
            for i, (membar_i, utilsbar_i) in enumerate(zip(membar, utilsbar)):
                m, u = _get_gpu_monitor(i)
                membar_i.value, utilsbar_i.value = m, u
                membar_i.description, utilsbar_i.description = 'M' + str(i) + ': ' + str(
                    m
                ) + '%', 'U' + str(i) + ': ' + str(u) + '%'
            time.sleep(1)


class Liveplot:
    def __init__(self, train_iteration, val_iteration, _plot = False, _enable_full_plot = True):
        self._plot = _plot
        self._enable_full_plot = _enable_full_plot
        self.history = hl.History()
        self.text_table = Texttable(max_width = 0)  #unlimited
        self.text_table.set_precision(4)
        self.train_timer = Timer(train_iteration)
        if val_iteration:
            self.val_timer = Timer(val_iteration)
        self.redis = redis.Redis()
        self.redis.set('progress', 0)
        self.redis.set('desc', '')
        self.redis.set('stage', 'stop')
        self.redis.set(
            'history',
            pickle.dumps({
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            })
        )

        # Label + Pregress
        self.train_progress = Output()
        self.train_progress_label = Output()
        self.val_progress = Output()
        self.val_progress_label = Output()
        display(self.train_progress_label)
        display(self.train_progress)
        display(self.val_progress_label)
        display(self.val_progress)
        # Train
        with self.train_progress:
            self.train_progress_bar = IntProgress(bar_style = 'info')
            self.train_progress_bar.min = 0
            self.train_progress_bar.max = train_iteration
            display(self.train_progress_bar)
        with self.train_progress_label:
            self.train_progress_label_text = Label(value = "Initialization")
            display(self.train_progress_label_text)
        # Validate
        with self.val_progress:
            self.val_progress_bar = IntProgress(bar_style = 'warning')
            self.val_progress_bar.min = 0
            self.val_progress_bar.max = 1 if val_iteration is None else val_iteration
            display(self.val_progress_bar)
        with self.val_progress_label:
            self.val_progress_label_text = Label(value = "Initialization")
            display(self.val_progress_label_text)

        # Plots
        if self._plot:
            # 4 chartplots
            self.loss_plot = Output()
            self.matrix_plot = Output()
            display(HBox([self.loss_plot, self.matrix_plot]))
            self.lr_plot = Output()
            self.norm_plot = Output()
            display(HBox([self.lr_plot, self.norm_plot]))

            # Canvas
            self.loss_canvas = hl.Canvas()
            self.matrix_canvas = hl.Canvas()
            self.lr_canvas = hl.Canvas()
            self.norm_canvas = hl.Canvas()

        # Memory
        if self._enable_full_plot:
            gpu_count = nvmlDeviceGetCount()
            total_bars = [Output() for _ in range(2 * gpu_count)]
            self.gpu_mem_monitor = total_bars[::2]
            self.gpu_utils_monitor = total_bars[1::2]
            display(HBox(total_bars))
            self.gpu_mem_monitor_bar = []
            self.gpu_utils_monitor_bar = []
            for i, (membar, utilsbar) in enumerate(zip(self.gpu_mem_monitor, self.gpu_utils_monitor)):
                with membar:
                    self.gpu_mem_monitor_bar.append(
                        IntProgress(orientation = 'vertical', bar_style = 'success')
                    )
                    self.gpu_mem_monitor_bar[-1].description = 'M' + str(i) + ': 0%'
                    self.gpu_mem_monitor_bar[-1].min = 0
                    self.gpu_mem_monitor_bar[-1].max = 100
                    display(self.gpu_mem_monitor_bar[-1])

                with utilsbar:
                    self.gpu_utils_monitor_bar.append(
                        IntProgress(orientation = 'vertical', bar_style = 'success')
                    )
                    self.gpu_utils_monitor_bar[-1].description = 'U' + str(i) + ': 0%'
                    self.gpu_utils_monitor_bar[-1].min = 0
                    self.gpu_utils_monitor_bar[-1].max = 100
                    display(self.gpu_utils_monitor_bar[-1])


            # Customize
            self.custom_train_output = Output()
            self.custom_val_output = Output()
            display(HBox([self.custom_train_output, self.custom_val_output]))

            # Log
            self.text_log = Output()
            display(self.text_log)


            # Start monitor thread
            global _STOP_GPU_MONITOR_
            _STOP_GPU_MONITOR_ = False
            self.thread = threading.Thread(
                target = _gpu_monitor_worker,
                args = (self.gpu_mem_monitor_bar, self.gpu_utils_monitor_bar)
            )
            self.thread.start()

    def _process_num(self, var):
        if var is None:
            return 'None'
        else:
            return f'{var:.4f}'
        
    def update_train_desc(self, epoch, batch_idx, loss_record, monitor_record, track_best):
        loss_record /= batch_idx
        metrics_collection = [
            ', '.join([
                '%s: %.6s' % (key.capitalize(), float(value) / batch_idx)
                for (key, value) in monitor_record.items()
            ])
        ]
        metrics_collection = ''.join(metrics_collection)
        misc, progress = self.train_timer.output(batch_idx)
        self.cache = (epoch, loss_record, metrics_collection, track_best, misc)
        self.train_progress_bar.value = batch_idx
        self.train_progress_label_text.value = f'Epoch: {epoch}, Loss: {self._process_num(loss_record)}, {metrics_collection}, Track best: {self._process_num(track_best)}, {misc}'
#         self.redis.set('desc', self.train_progress_label_text.value)
#         self.redis.set('progress', progress)
        
    
    def update_val_desc(self, epoch, batch_idx, loss_record, monitor_record, track_best):
        loss_record /= batch_idx
        metrics_collection = [
            ', '.join([
                '%s: %.6s' % (key.capitalize(), float(value) / batch_idx)
                for (key, value) in monitor_record.items()
            ])
        ]
        metrics_collection = ''.join(metrics_collection)
        misc, progress = self.val_timer.output(batch_idx)
        self.cache = (epoch, loss_record, metrics_collection, track_best, misc)
        self.val_progress_bar.value = batch_idx
        self.val_progress_label_text.value = f'Epoch: {epoch}, Loss: {self._process_num(loss_record)}, {metrics_collection}, Track best: {self._process_num(track_best)}, {misc}'
#         self.redis.set('desc', self.val_progress_label_text.value)
#         self.redis.set('progress', progress)
    
    
    def cali_desc(self, track_best):
        (epoch, loss_record, metrics_collection, _, misc) = self.cache
        self.train_progress_label_text.value = f'Epoch: {epoch}, Loss: {self._process_num(loss_record)}, {metrics_collection}, Track best: {self._process_num(track_best)}, {misc}'
#         self.redis.set('desc', self.train_progress_label_text.value)
        

    def record(self, epoch, key, value, mode = ''):
        self.history.log(epoch, **{key: value})
        
        
    def show_grid(self, phase, tensor):
        if self._enable_full_plot:
            if phase == 'train':
                out_stream = self.custom_train_output
            else:
                out_stream = self.custom_val_output

            with out_stream:
                _display.clear_output(wait = True)
                for i in tensor:
                    if type(i) == tuple or type(i) == list:
                        i, aux = i
                        aux = aux.detach().cpu().numpy()
                    else:
                        aux = np.zeros(i.size(0))

                    if i.dim() < 3 or (i.shape[1] != 1 and i.shape[1] != 3):
                        data = i.detach().cpu().float().reshape(i.size(0), -1).numpy()
                        pca = PCA(2)
                        r = pca.fit_transform(data)
                        plt.figure(figsize=(5, 5))
                        plt.scatter(r[:, 0], r[:, 1], c = aux)
                        plt.grid()
                        plt.tight_layout()
                        plt.title(f'Explained Variance Ratio: {sum(pca.explained_variance_ratio_)}')
                        plt.show()
                    else:
                        unique_index = np.unique(aux)
                        if unique_index.any() != 0:
                            groups = len(unique_index)
                            for index in unique_index:
                                subgroup = i[aux==index].detach()[:8]
                                img = torchvision.utils.make_grid(subgroup, len(subgroup), 0)
                                npimg = img.cpu().float().numpy()
                                npimg += 1
                                npimg /= 2
                                npimg = np.clip(npimg, 0, 1)
#                                 if npimg.min() >= -1 and npimg.max <= 1:
#                                     npimg += 1
#                                     npimg /= 2
#                                     npimg = np.clip()
#                                 else:
#                                     npimg -= npimg.min()
#                                     npimg /= npimg.max()
                                plt.figure(figsize=((5*len(subgroup))/6, 5))
                                plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
                                plt.axis('off')
                                plt.tight_layout()
                                plt.show()
                        else:
                            img = torchvision.utils.make_grid(i[:64].detach(), 8, 0)
                            npimg = img.cpu().float().numpy()
                            npimg += 1
                            npimg /= 2
                            npimg = np.clip(npimg, 0, 1)
#                             if npimg.min() >= -1 and npimg.max <= 1:
#                                 npimg += 1
#                                 npimg /= 2
#                             else:
#                                 npimg -= npimg.min()
#                                 npimg /= npimg.max()
                            plt.figure(figsize=(5, 5))
                            plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
                            plt.axis('off')
                            plt.tight_layout()
                            plt.show()

            
    def plot(self, epoch, history_, epoch_str):
        if self._enable_full_plot:
#             self.redis.set('history', pickle.dumps(history_.records))
            if self._plot:
                with self.loss_plot:
                    self.loss_canvas.draw_plot([self.history["train_loss"], self.history['val_loss']])

                with self.matrix_plot:
                    self.matrix_canvas.draw_plot([self.history['train_acc'], self.history['val_acc']])

                with self.lr_plot:
                    self.lr_canvas.draw_plot([self.history['lr']])

                with self.norm_plot:
                    self.norm_canvas.draw_plot([self.history['norm']])

            with self.text_log:
                if epoch == 1:
                    self.text_table.add_row(['Epochs'] + list(history_.records.keys()))
                    self.text_table.set_cols_width(
                        [6] + [len(format(i[-1], '.8f')) for i in list(history_.records.values())]
                    )
                self.text_table.add_row(
                    [epoch_str] + [format(i[-1], '.4f') for i in list(history_.records.values())]
                )
                _display.clear_output(wait = True)
                print(self.text_table.draw())

        
