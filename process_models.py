import numpy as np
from torch import nn
from torch.optim import Adam
import torch


class ModelProcessor(object):
    def __init__(self,
                 model,
                 train_loader,
                 test_loader,
                 evaluator=None,
                 feature_extractor=None):
        assert model
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_optimizer = Adam(self.model.parameters(), lr=1e-3)
        scheduler_params = dict(
            mode='min',
            factor=0.5,
            patience=1,
            verbose=True,
            threshold=0.001,
            threshold_mode='abs',
            cooldown=0,
            min_lr=1e-6,
            eps=1e-08
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.model_optimizer, **scheduler_params)
        self.model = model.float()
        self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.weights_init(self.model)
        self.evaluator = evaluator
        self.feature_extractor = feature_extractor

    @staticmethod
    def weights_init(m):
        class_name = m.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif class_name.find('Linear') != -1:
            weight_shape = list(m.weight.data.size())
            fan_in = weight_shape[1]
            fan_out = weight_shape[0]
            w_bound = np.sqrt(6. / (fan_in + fan_out))
            m.weight.data.uniform_(-w_bound, w_bound)
            m.bias.data.fill_(0)

    def train_model(self, epoch=50, save_best_model=True, model_path=None):
        assert self.model
        for epoch in range(epoch):
            if self.evaluator:
                self.evaluator.next_epoch()
            self.model.train()
            for data in self.train_loader:
                self.model_optimizer.zero_grad()
                inputs, target, file_, _ = data
                target = target.to(self.device)
                output = self.model(inputs)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.model_optimizer.step()
                if self.evaluator:
                    self.evaluator.log_train_iter(output, target, loss)

            self.model.eval()
            total_loss = 0
            for data in self.test_loader:
                inputs, target, file_, _ = data
                target = target.to(self.device)
                with torch.no_grad():
                    output = self.model(inputs)
                    loss = self.loss_fn(output, target)
                    total_loss += loss.item()

                if self.evaluator:
                    self.evaluator.log_eval_iter(output, target, loss)
            total_loss /= len(self.test_loader)
            self.scheduler.step(total_loss)
            if self.evaluator:
                self.evaluator.print_epoch_metrics()

                if save_best_model:
                    assert model_path
                    if self.evaluator.is_best_model():
                        self.save_model(model_path=model_path)

    def save_model(self, model_path):
        assert model_path
        torch.save(self.model.state_dict(), model_path)
        print('Model saved...')

    def load_model(self, model_path):
        assert model_path
        assert self.model
        self.model.load_state_dict(torch.load(model_path))
        print('Model loaded...')

    def test_model(self):
        assert self.model
        self.model.eval()
        for data in self.test_loader:
            if self.feature_extractor:
                self.feature_extractor.connect()
            inputs, target, file_, o_index = data
            with torch.no_grad():
                target = target.to(self.device)
                output = self.model(inputs)

            if self.evaluator:
                self.evaluator.log_test_iter(output,
                                             target,
                                             file_,
                                             feature_extractor=self.feature_extractor,
                                             input_data_index=o_index)

            if self.feature_extractor:
                self.feature_extractor.remove()

        if self.evaluator:
            self.evaluator.print_eval_reports()
