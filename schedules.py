# coding=utf-8
# Copyright 2021 Intel Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from timeit import default_timer as get_now
from torch.optim.lr_scheduler import LambdaLR, ConstantLR, ReduceLROnPlateau


def build_scheduler(optimizer, schedulers, curver):
    def get_warmup_calc(current_step: int, scheduler, curver):
        position = scheduler.get_correct_position(current_step)
        if scheduler.still_in_warmup(position):
            warmup_position = scheduler.get_warmup_percent(position)
            return curver.get_warmup(warmup_position)
        else:
            return curver.get_decay(scheduler.get_total(), position, scheduler.get_total_warmup())

    # if type(scheduler) == list:
    lr_lambda = [lambda x: get_warmup_calc(x, scheduler=schedulers[0],curver=curver[0]),
    lambda x: get_warmup_calc(x, scheduler=schedulers[1], curver=curver[1]),
    lambda x: get_warmup_calc(x, scheduler=schedulers[2], curver=curver[2])]
 

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)


class LinearCurve:
    def get_warmup(self, value):
        return value

    def get_decay(self, total, current, total_warmup):
        return max(0.0, total - current) / max(1.0, total - total_warmup)

class ConstantCurve:
    def get_warmup(self, value):
        return value

    def get_decay(self, total, current, total_warmup):
        return 1.0

class ExpCurve:
    def __init__(self, schedule_args):
        self.schedule_args = schedule_args

    def get_warmup(self, value):
        return value ** 2

    def get_decay(self, current, total_warmup):
        return self.schedule_args.decay_rate ** (
            (current - total_warmup) / self.schedule_args.decay_step
        )


class BaseScheduler:
    def __init__(self, schedule_args, warmup_proportion):
        self.schedule_args = schedule_args
        self.warmup_proportion = warmup_proportion
        # print('self warmup',self.warmup_proportion)

    def still_in_warmup(self, position):
        # print('position',position)
        # print('total warmup', self.get_total_warmup())

        return position < self.get_total_warmup()

    def get_total_warmup(self):
        # print('get warmup proportion', self.warmup_proportion)
        if self.warmup_proportion < 1:
            return self.get_total() * self.warmup_proportion
        return self.warmup_proportion 

    def get_warmup_percent(self, position):
        return position / self.get_total_warmup()


class StepScheduler(BaseScheduler):
    def __init__(self, schedule_args, warmup_proportion, extra_args):
        super().__init__(schedule_args, warmup_proportion)
        self.extra_args = extra_args
        # print('warmup_proportion',self.warmup_proportion)

    def get_correct_position(self, current_step):
        return current_step

    def get_total(self):
        return self.extra_args.max_steps


class FixedWarmupScheduler(StepScheduler):
    def __init__(self, schedule_args, extra_args):
        super().__init__(schedule_args, extra_args)

    def get_total_warmup(self):
        return self.schedule_args.num_warmup_steps


class TimeScheduler(BaseScheduler):
    def __init__(self, schedule_args, warmup_proportion, extra_args):
        super().__init__(schedule_args, warmup_proportion)
        self.extra_args = extra_args

    def get_correct_position(self, current_step):
        return (get_now() - self.extra_args.exp_start_marker) / 3600

    def get_total(self):
        return self.extra_args.total_training_time


CURVES = {"linear": lambda args: LinearCurve(), "exp": lambda args: ExpCurve(args), "constant": lambda args: ConstantCurve()}

SCHEDULES = {"step": StepScheduler, "constant_step": FixedWarmupScheduler, "time": TimeScheduler}


def get_scheduler(schedule_args, optimizer, extra_args):
    # curver = CURVES[schedule_args.curve](schedule_args)
    schedulers = []
    curvers = []
    if schedule_args.warmup_proportion_list:
        warmup_steps = schedule_args.warmup_proportion_list
    elif schedule_args.num_warmup_steps:
        warmup_steps = schedule_args.num_warmup_steps
    for ix, sch in enumerate(schedule_args.lr_schedule):
        schedulers.append(SCHEDULES[sch](schedule_args, warmup_steps[ix], extra_args))
    curvers = [CURVES[cur](schedule_args) for _, cur in enumerate(schedule_args.curve)]
    return build_scheduler(optimizer, schedulers, curvers)