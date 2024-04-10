import os
from datetime import datetime

import torch


class InfIterator:
    def __init__(self, iterable):
        self.iterable = iterable
        self.iterator = iter(self.iterable)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.iterable)
            return next(self.iterator)

class Logger:
    def __init__(
        self,
        save_dir=None,
        save_only_last=True,
        print_every=100,
        save_every=100,
        total_step=0,
        print_to_stdout=True,
    ):
        if save_dir is not None:
            self.save_dir = save_dir
            os.makedirs(self.save_dir, exist_ok=True)
        else:
            self.save_dir = None

        self.print_every = print_every
        self.save_every = save_every
        self.save_only_last = save_only_last
        self.step_count = 0
        self.total_step = total_step
        self.print_to_stdout = print_to_stdout

        self.writer = None
        self.start_time = None
        self.groups = dict()
        self.models_to_save = dict()
        self.objects_to_save = dict()

    def register_model_to_save(self, model, name):
        assert name not in self.models_to_save.keys(), "Name is already registered."

        self.models_to_save[name] = model

    def register_object_to_save(self, object, name):
        assert name not in self.objects_to_save.keys(), "Name is already registered."

        self.objects_to_save[name] = object

    def step(self):
        self.step_count += 1
        if self.step_count % self.print_every == 0:
            if self.print_to_stdout:
                self.print_log(self.step_count, self.total_step, elapsed_time=datetime.now() - self.start_time)

        if self.step_count % self.save_every == 0:
            if self.save_only_last:
                self.save_models()
                self.save_objects()
            else:
                self.save_models(self.step_count)
                self.save_objects(self.step_count)

    def meter(self, group_name, log_name, value):
        if group_name not in self.groups.keys():
            self.groups[group_name] = dict()

        if log_name not in self.groups[group_name].keys():
            self.groups[group_name][log_name] = Accumulator()

        self.groups[group_name][log_name].update_state(value)

    def reset_state(self):
        for _, group in self.groups.items():
            for _, log in group.items():
                log.reset_state()

    def print_log(self, step, total_step, elapsed_time=None):
        print(f"[Step {step:5d}/{total_step}]", end="  ")

        for name, group in self.groups.items():
            print(f"({name})", end="  ")
            for log_name, log in group.items():
                res = log.result()
                if res is None:
                    continue

                if "acc" in log_name.lower():
                    print(f"{log_name} {res:.2f}", end=" | ")
                else:
                    print(f"{log_name} {res:.4f}", end=" | ")

        if elapsed_time is not None:
            print(f"(Elapsed time) {elapsed_time}")
        else:
            print()

    def save_models(self, suffix=None):
        if self.save_dir is None:
            return
        for name, model in self.models_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(model.state_dict(), os.path.join(self.save_dir, f"{_name}.pt"))

            if self.print_to_stdout:
                print(f"{name} is saved to {self.save_dir}")

    def save_objects(self, suffix=None):
        if self.save_dir is None:
            return

        for name, obj in self.objects_to_save.items():
            _name = name
            if suffix:
                _name += f"_{suffix}"
            torch.save(obj, os.path.join(self.save_dir, f"{_name}.pt"))

            if self.print_to_stdout:
                print(f"{name} is saved to {self.save_dir}")

    def start(self):
        if self.print_to_stdout:
            print("Training starts!")
        self.start_time = datetime.now()

    def finish(self):
        if self.step_count % self.save_every != 0:
            if self.save_only_last:
                self.save_models()
                self.save_objects()
            else:
                self.save_models(self.step_count)
                self.save_objects(self.step_count)

        if self.print_to_stdout:
            print("Training is finished!")

class Accumulator:
    def __init__(self):
        self.data = 0
        self.num_data = 0

    def reset_state(self):
        self.data = 0
        self.num_data = 0

    def update_state(self, tensor):
        with torch.no_grad():
            self.data += tensor
            self.num_data += 1

    def result(self):
        if self.num_data == 0:
            return None
        data = self.data.item() if hasattr(self.data, 'item') else self.data
        return float(data) / self.num_data
