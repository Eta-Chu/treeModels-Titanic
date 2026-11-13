import numpy as np
from copy import deepcopy
from sklearn.metrics import accuracy_score
from scipy.special import expit

def sigmoid(x):
    return expit(x)

# def sigmoid(x):
# #     return 1 / (1 + np.exp(-x))
#     return np.exp(x) / (1 + np.exp(x))

def binary_cross(y_true: np.ndarray, F: np.ndarray):
    return np.mean(- F * y_true + np.log(1 + np.exp(F)))

class DecisionTree:

    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth
        if self.max_depth != 1:
            raise NotImplementedError('Currently noly depth=1 is supported!')
        self.optimal_f = None
        self.threshold = None
        self.split_gain = 0.0
        self.left_value = None
        self.right_value = None

    def __cal_gain(self, 
                   leftGi: float, 
                   leftHi: float, 
                   rightGi: float, 
                   rightHi: float, 
                   lamda: float, 
                   gamma: float):
        
        parentGi = leftGi + rightGi
        parentHi = leftHi + rightHi
        split_gain = (1 / 2) * ((leftGi ** 2) / (leftHi + lamda) + (rightGi ** 2) / (rightHi + lamda) - (parentGi ** 2) / (parentHi + lamda)) - gamma
        
        return split_gain
        
    def fit(self, 
            x_train: np.ndarray, 
            gi: np.ndarray, 
            hi: np.ndarray, 
            lamda: float, 
            gamma: float):

        self.optimal_f = None
        self.threshold = None
        self.split_gain = 0.0
        
        # greedy search
        candidate = []
        sample_num, feature_num = x_train.shape
        for i in range(feature_num):
            chosen_feature = x_train[:, i]
            sorted_idx = np.argsort(chosen_feature)
            sorted_feature = chosen_feature[sorted_idx]
            sorted_Gi = np.cumsum(gi[sorted_idx])
            sorted_Hi = np.cumsum(hi[sorted_idx])

            split_idx = 1
            while split_idx < sample_num:
                if sorted_feature[split_idx - 1] == sorted_feature[split_idx]:
                    pass
                else:
                    threshold = (sorted_feature[split_idx - 1] + sorted_feature[split_idx]) / 2
                    leftGi = sorted_Gi[split_idx - 1]
                    leftHi = sorted_Hi[split_idx - 1]
                    rightGi = sorted_Gi[-1] - leftGi
                    rightHi = sorted_Hi[-1] - leftHi
                    split_gain = self.__cal_gain(leftGi, leftHi, rightGi, rightHi, lamda, gamma)
                    left_value = - leftGi / (leftHi + lamda)
                    right_value = - rightGi / (rightHi + lamda)
                    candidate.append([i, threshold, split_gain, left_value, right_value])
                split_idx += 1
        candidate = np.array(candidate)
        best_split = np.argmax(candidate[:, 2])
        self.optimal_f, self.threshold, self.split_gain, self.left_value, self.right_value = candidate[best_split]
        self.optimal_f = int(self.optimal_f)
        
    def predict(self, x_predict: np.ndarray):
        sample_num, feature_num = x_predict.shape
        y_predict = np.full(shape=(sample_num, ), fill_value=self.right_value)
        left_idx = np.where(x_predict[:, self.optimal_f] < self.threshold)[0]
        y_predict[left_idx] = self.left_value
        return y_predict
    
class XGBoost:
    
    def __init__(self, lr: float, gamma: float, lamda: float):
        self.lr = lr
        self.gamma = gamma
        self.lamda = lamda
        self.samples_num = 0
        self.F0 = None
        self.F = None
        self.F_val = None
        self.forest = []
        self.log = {}
        self.log['train_loss'] = []
        self.log['train_acc'] = []
        self.log['val_loss'] = []
        self.log['val_acc'] = []
    
    def __initlize(self, y_train: np.ndarray):
        # training first tree
        p0 = np.mean(y_train)
        F0 = np.log(p0 / (1 - p0))
        F = np.ones(shape=(self.samples_num, )) * F0
        self.F0 = F
    
    def cal_gradient(self, y_train: np.ndarray, y_predict: np.ndarray):
        prob = sigmoid(y_predict)
        gi = prob - y_train
        hi = prob * (1 - prob)
        return gi, hi

    def run(self, 
            x_train: np.ndarray, 
            y_train: np.ndarray, 
            x_val: np.ndarray, 
            y_val: np.ndarray, 
            iteration: int):
        
        self.samples_num = x_train.shape[0]
        
        self.__initlize(y_train)
        self.F = np.copy(self.F0)
        self.F_val = np.full_like(y_val, fill_value=self.F[0], dtype=np.float64)

        for i in range(iteration):
            print(f'start {i}th iteration!')
            gi, hi = self.cal_gradient(y_train, self.F)
            tree = DecisionTree(max_depth=1)
            tree.fit(x_train, gi, hi, self.lamda, self.gamma)
            self.forest.append(deepcopy(tree))
            ft = tree.predict(x_train)
            self.F += self.lr * ft

            # log
            train_prob = sigmoid(self.F)
            train_pred = train_prob > 0.5
            train_acc = accuracy_score(y_train, train_pred)
            train_loss = binary_cross(y_train, self.F)
            ft_val = tree.predict(x_val)
            self.F_val += self.lr * ft_val
            val_prob = sigmoid(self.F_val)
            val_pred = val_prob > 0.5
            val_acc = accuracy_score(y_val, val_pred)
            val_loss = binary_cross(y_val, self.F_val)
            self.log['train_loss'].append([i, train_loss])
            self.log['train_acc'].append([i, train_acc])
            self.log['val_loss'].append([i, val_loss])
            self.log['val_acc'].append([i, val_acc])

    def test(self, x_test: np.ndarray):
        F = np.full(shape=(x_test.shape[0], ), fill_value=self.F0[0])
        for tree in self.forest:
            F += self.lr * tree.predict(x_test)
        test_prob = sigmoid(F)
        test_predict = test_prob > 0.5
        return test_predict