from visdom import Visdom
import torch
from typing import Union, Optional
import numpy as np


class ImageShow:
    idx = 0
    max_idx = 20

    def __init__(self, env_name="main"):
        try:
            self.vis = Visdom()
        except:
            self.vis = None
        self.env = env_name

    def show_image(self, arr: Union[torch.Tensor, np.ndarray], env: Optional[str] = None):
        if self.vis is not None :
            if isinstance(arr, np.ndarray):
                arr = torch.from_numpy(arr)
            assert isinstance(arr, torch.Tensor)
            if arr.ndim == 3:
                arr: torch.Tensor = arr.permute(dims=(2, 0, 1))
                arr= arr.type(torch.float32)
            env = env if env is not None else self.env
            self.idx += 1
            self.vis.image(arr, f"image{self.idx % self.max_idx}", env)
        else:
            return

    def show_text(self, content: str):
        if self.vis is not None:
            self.vis.text(content, win="text", env=self.env, append=self.vis.win_exists("text"))
        else:
            return


    def plot(self, y, x, caption):
        if self.vis is not None:
            y = [i.cpu() for i in y]
            self.vis.line(y, [x], win=caption, opts={"legend": ["loss"]}, update="append")