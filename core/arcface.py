from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter

class ArcMarginProduct(nn.Module):
    def __init__(self, in_features=128, num_classes=200, s=48.0, m=0.50, easy_margin=False, class_to_superclass=None):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.weight = Parameter(torch.Tensor(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
        # init.kaiming_uniform_()
        # self.weight.data.normal_(std=0.001)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.class_to_superclass = None
        if class_to_superclass is not None:
            assert self.num_classes == len(class_to_superclass)
            assert self.num_classes == max([i for i in class_to_superclass]) + 1
            for cl, supcl in class_to_superclass.items():
                assert isinstance(cl, int) 
                assert cl >= 0
                assert cl < self.num_classes
                assert isinstance(supcl, int) 
                assert supcl >= 0
                assert supcl < self.num_classes

            superclass_list = [class_to_superclass[i] for i in range(0, self.num_classes)]
            print("using superclass softmax: {} (total {} superclasses) ...".format(superclass_list[:20], len(set(superclass_list))))
            for cl in superclass_list:
                assert isinstance(cl, int) 
            
            self.superclass_vector = nn.Parameter(
                torch.LongTensor(superclass_list), requires_grad=False)

            self.label_range = nn.Parameter(
                torch.arange(self.num_classes), requires_grad=False)
   
        print("using arcface with {} classes".format(self.num_classes))        
 
    def update_margin(self, m):
        assert m < 0.51
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, label):
        """
        features with of a same mutex_label (though of different labels), 
            do not compete, i.e., can be very similar but do not promote loss.
        """
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        if self.class_to_superclass is not None:
            super_label = self.superclass_vector[label.view(-1)]
            mutex_mask = (
                torch.eq(super_label.view(-1, 1), self.superclass_vector.view(1, -1)) * \
                torch.ne(label.view(-1, 1), self.label_range.view(1, -1))
            ).detach().float()
            assert mutex_mask.device == x.device
            cosine = cosine * (1 - mutex_mask) + cosine.mean(1, keepdim=True).detach() * mutex_mask
            
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output
