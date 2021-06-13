from pytorch_lightning import LightningDataModule

class ImageLevelData(LightningDataModule):

    def __init__(self, data_root: str, img_label_csv: str):
        super().__init__()

    def prepare_data(self):
        pass

    def setup(self, stage: str):  
        """[summary]

        Args:
            stage (str): trainer 调用的函数名
        """
        print(f"Stage: {stage}")

    def train_dataloader(self):
        pass

    def test_dataloader(self):
        pass