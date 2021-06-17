from visdom import Visdom
import torch
from typing import Union, Optional
import numpy as np


class ImageShow:

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
            env = env if env is not None else self.env
            self.vis.image(arr, "image", env)
        else:
            return

    def show_text(self, content: str):
        if self.vis is not None:
            self.vis.text(content, win="text", env=self.env, append=self.vis.win_exists("text"))
        else:
            return