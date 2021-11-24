from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--optimizer', default='Adam', type=str, help='adam or sgd for optimization')
        self.parser.add_argument('--lr', default=0.001, type=float, help='learning rate of Adam')
        self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[10000, 20000], help='steps to drop LR in training samples')
        self.parser.add_argument('--weight_decay', default=0.0001, type=float, help='weights regularizer')
        self.parser.add_argument('--test_on', default=True, type=bool, help='whether using test on')
        self.parser.add_argument('--ckpt', default='./ckpt', type=str, help='Path to save model')
        self.mode = 'train'