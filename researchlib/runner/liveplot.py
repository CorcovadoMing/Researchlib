from ..models import GANModel
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
                membar_i.description, utilsbar_i.description = 'M'+str(i), 'U'+str(i)
            break
        else:
            for i, (membar_i, utilsbar_i) in enumerate(zip(membar, utilsbar)):
                m, u = _get_gpu_monitor(i)
                membar_i.value, utilsbar_i.value = m, u
                membar_i.description, utilsbar_i.description = 'M'+str(i)+': ' + str(m) + '%', 'U'+str(i)+': ' + str(u) + '%'
            time.sleep(1)


def _list_avg(l):
    return sum(l) / len(l)


class Liveplot:

    def __init__(self, model, total_iteration, _plot=False):
        self._plot = _plot
        self._gan = True if type(model) == GANModel else False
        self.history = hl.History()
        self.text_table = Texttable(max_width=0)  #unlimited
        self.timer = Timer(total_iteration)
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
            }))

        # Label + Pregress
        self.progress = Output()
        self.progress_label = Output()
        display(HBox([self.progress_label, self.progress]))
        with self.progress:
            self.progress_bar = IntProgress(bar_style='info')
            self.progress_bar.min = 0
            self.progress_bar.max = total_iteration
            display(self.progress_bar)
        with self.progress_label:
            self.progress_label_text = Label(value="Initialization")
            display(self.progress_label_text)

        if self._plot:
            # 4 chartplots
            self.loss_plot = Output()
            self.matrix_plot = Output()
            display(HBox([self.loss_plot, self.matrix_plot]))
            self.lr_plot = Output()
            self.norm_plot = Output()
            display(HBox([self.lr_plot, self.norm_plot]))

            # GAN visualization
            if self._gan:
                self.gan_gallary = Output()
                display(self.gan_gallary)
                self.gan_canvas = hl.Canvas()

            # Canvas
            self.loss_canvas = hl.Canvas()
            self.matrix_canvas = hl.Canvas()
            self.lr_canvas = hl.Canvas()
            self.norm_canvas = hl.Canvas()

        # Memory and Log
        gpu_count = nvmlDeviceGetCount()
        total_bars = [Output() for _ in range(2*gpu_count)] 
        self.gpu_mem_monitor = total_bars[::2]
        self.gpu_utils_monitor = total_bars[1::2]
        self.text_log = Output()
        display(HBox(total_bars))
        display(self.text_log)

        self.gpu_mem_monitor_bar = []
        self.gpu_utils_monitor_bar = []
        for i, (membar, utilsbar) in enumerate(zip(self.gpu_mem_monitor, self.gpu_utils_monitor)):
            with membar:
                self.gpu_mem_monitor_bar.append(IntProgress(orientation='vertical', bar_style='success'))
                self.gpu_mem_monitor_bar[-1].description = 'M'+str(i)+': 0%'
                self.gpu_mem_monitor_bar[-1].min = 0
                self.gpu_mem_monitor_bar[-1].max = 100
                display(self.gpu_mem_monitor_bar[-1])

            with utilsbar:
                self.gpu_utils_monitor_bar.append(IntProgress(orientation='vertical', bar_style='success'))
                self.gpu_utils_monitor_bar[-1].description = 'U'+str(i)+': 0%'
                self.gpu_utils_monitor_bar[-1].min = 0
                self.gpu_utils_monitor_bar[-1].max = 100
                display(self.gpu_utils_monitor_bar[-1])

        # Start monitor thread
        global _STOP_GPU_MONITOR_
        _STOP_GPU_MONITOR_ = False
        self.thread = threading.Thread(
            target=_gpu_monitor_worker,
            args=(self.gpu_mem_monitor_bar, self.gpu_utils_monitor_bar))
        self.thread.start()

    def update_progressbar(self, value):
        self.progress_bar.value = value

    def update_loss_desc(self, epoch, g_loss_history, d_loss_history,
                         loss_history):
        misc, progress = self.timer.output()
        if self._gan:
            self.progress_label_text.value = f'Epoch: {epoch}, G Loss: {_list_avg(g_loss_history):.4f}, D Loss: {_list_avg(d_loss_history):.4f}, {misc}'
        else:
            self.progress_label_text.value = f'Epoch: {epoch}, Loss: {_list_avg(loss_history):.4f}, {misc}'
        self.redis.set('desc', self.progress_label_text.value)
        self.redis.set('progress', progress)

    def record(self, epoch, key, value, mode=''):
        if mode == 'gan' and self._gan == True:
            self.history.log(epoch, **{key: value})
        elif mode == 'non-gan' and self._gan == False:
            self.history.log(epoch, **{key: value})
        else:
            self.history.log(epoch, **{key: value})

    def plot(self, epoch, history_, epoch_str):
        self.redis.set('history', pickle.dumps(history_.records))
        if self._plot:
            if self._gan:
                with self.gan_gallary:
                    self.gan_canvas.draw_image(self.history['image'])

            with self.loss_plot:
                if self._gan:
                    self.loss_canvas.draw_plot([
                        self.history["train_g_loss"],
                        self.history['train_d_loss']
                    ])
                else:
                    self.loss_canvas.draw_plot(
                        [self.history["train_loss"], self.history['val_loss']])

            with self.matrix_plot:
                if self._gan:
                    self.matrix_canvas.draw_plot(
                        [self.history['inception_score'], self.history['fid']])
                else:
                    self.matrix_canvas.draw_plot(
                        [self.history['train_acc'], self.history['val_acc']])

            with self.lr_plot:
                if self._gan:
                    self.lr_canvas.draw_plot(
                        [self.history['g_lr'], self.history['d_lr']])
                else:
                    self.lr_canvas.draw_plot([self.history['lr']])

            with self.norm_plot:
                self.norm_canvas.draw_plot([self.history['norm']])

        with self.text_log:
            if epoch == 1:
                self.text_table.add_row(['Epochs'] +
                                        list(history_.records.keys())[:-1])
                self.text_table.set_cols_width([
                    6,
                ] + [
                    len(format(i[-1], '.4f'))
                    for i in list(history_.records.values())[:-1]
                ])
            self.text_table.add_row([epoch_str] + [
                format(i[-1], '.4f')
                for i in list(history_.records.values())[:-1]
            ])
            _display.clear_output(wait=True)
            print(self.text_table.draw())
