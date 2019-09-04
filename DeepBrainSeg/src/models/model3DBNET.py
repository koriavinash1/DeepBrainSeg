import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict



class Projection_Layer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate=0, pad = 0):
        super(Projection_Layer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features,num_output_features , kernel_size=1, stride=1, bias=False)),
        self.drop_rate = drop_rate
        self.pad = pad

    def forward(self, x):
        new_features = super(Projection_Layer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        p3d = (-2,-2,-2, -2, -2, -2)

        if not self.pad: new_features = F.pad(new_features, p3d)
        else : new_features = new_features

        return new_features


class _Layer(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, pad = 0):
        super(_Layer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True))
        self.add_module('conv1', nn.Conv3d(num_input_features,num_output_features , kernel_size=3, stride=1, bias=False, padding = pad)),
        self.drop_rate = drop_rate


    def forward(self, x):
        new_features = super(_Layer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features



class BLOCK(nn.Module):
    def __init__(self, num_input_features, num_output_features, drop_rate=0, pad = 0):
        super(BLOCK, self).__init__()
        self.pad = pad

        self.layer1 = _Layer(num_input_features,num_output_features,drop_rate, pad)
        self.layer2 = _Layer(num_output_features,num_output_features,drop_rate, pad)
        self.pro = Projection_Layer(num_input_features, num_output_features, pad = self.pad)
        self.num_output_features = num_output_features

    def forward(self, x):
        new_features = self.layer2(self.layer1(x))
        num_input_features = x.size()[1]
        projection_op = self.pro(x)
        return torch.add(new_features, projection_op)


# class BLOCK(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features, drop_rate, pad = 0):
#         super(BLOCK, self).__init__()
#         self.pad = pad
#         self.add_module('layer1', _Layer(num_input_features,num_output_features,drop_rate, pad)),
#         self.add_module('layer2', _Layer(num_output_features,num_output_features,drop_rate, pad)),
#         self.proj = Projection_Layer(num_input_features, num_output_features, pad = pad)

#     def forward(self, x):
#         new_features = super(BLOCK, self).forward(x)
#         # num_input_features = x.size()[1]
#         projection_op = self.proj(x)
#         return torch.add(new_features, projection_op)



class Residual_3D_Net(nn.Sequential):
    def __init__(self,in_channels= 4, convs=[30, 40, 40, 50]):
        super(Residual_3D_Net, self).__init__()
        self.add_module('resblock 1', _Layer(in_channels, convs[0], 0.2))
        self.add_module('resblock 2', _Layer(convs[0], convs[0], 0.2))
        for i in range(3):
            self.add_module('resblock %d' % (i+3), BLOCK(convs[i], convs[i+1], 0.2))

    def forward(self,x):
        new_features =super(Residual_3D_Net, self).forward(x)
        return new_features



class BrainNet_3D(nn.Module):
    def __init__(self, in_channels= 4, convs=[30, 40, 40, 50, 150, 5]):
        super(BrainNet_3D, self).__init__()
        self.high_res_net = Residual_3D_Net(in_channels, convs)
        self.low_res_net  = Residual_3D_Net(in_channels, convs)

        self.features     = nn.Sequential(OrderedDict([]))
        self.features.add_module('final_block', BLOCK(2*convs[3], convs[4], 0.2, pad = 1))
        self.features.add_module('final_layer', _Layer(convs[4], convs[5], 0.2, pad = 1))

    def forward(self, high_res_vol, low_res_vol):
        high_res_features = self.high_res_net(high_res_vol)
        low_res_features  = self.low_res_net(low_res_vol)
        # print ('high_res_features',high_res_features.size())
        # print ('low_res_features',low_res_features.size())
        size = high_res_features.size()
        #Upsample low_res_features
        low_res_upsample_features = F.upsample(low_res_features,(9,9,9))
        # print ('low_res_upsample_features',low_res_upsample_features.size())
        concat = torch.cat([high_res_features, low_res_upsample_features], 1)
        concat = self.features(concat)
        return concat

# ------------------------------------------------------------------------------------------------

class _Layer_Inception(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, pad = 0):
        super(_Layer_Inception, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)

        self.kernel_3x3 = nn.Conv3d(num_input_features, num_output_features//3, kernel_size=3, stride=1, bias=False, padding = (1 if pad else 0))
        self.kernel_5x5 = nn.Conv3d(num_input_features, num_output_features//3, kernel_size=5, stride=1, bias=False, padding = (2 if pad else 1))
        self.kernel_7x7 = nn.Conv3d(num_input_features, (num_output_features - 2*(num_output_features//3)), kernel_size=7, stride=1, bias=False, padding = (3 if pad else 2))
        self.drop_rate = drop_rate
        self.pad = pad

        # self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        # self.add_module('relu1', nn.ReLU(inplace=True))
        # self.add_module('conv1', nn.Conv3d(num_input_features,num_output_features , kernel_size=3, stride=1, bias=False, padding = pad)),
        # self.drop_rate = drop_rate


    def forward(self, x):
        new_features = self.relu(self.norm(x))
        new_features = torch.cat([self.kernel_3x3(new_features), self.kernel_5x5(new_features), self.kernel_7x7(new_features)], 1)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)

        return new_features

class BLOCK_Inception(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, drop_rate, pad = 0):
        super(BLOCK_Inception, self).__init__()
        self.layer1 = _Layer_Inception(num_input_features,num_output_features,drop_rate, pad)
        self.layer2 = _Layer_Inception(num_output_features,num_output_features,drop_rate, pad)
        self.pro = Projection_Layer(num_input_features, num_output_features, pad = pad)

    def forward(self, x):
        new_features = self.layer2(self.layer1(x))
        projection_op = self.pro(x)
        return torch.add(new_features, projection_op)


class Residual_3D_Net_Inception(nn.Sequential):
    def __init__(self,in_channels= 4, convs=[30, 40, 40, 50]):
        super(Residual_3D_Net_Inception, self).__init__()
        self.add_module('resblock 1', _Layer_Inception(in_channels, convs[0], 0.2))
        self.add_module('resblock 2', _Layer_Inception(convs[0], convs[0], 0.2))
        for i in range(3):
            self.add_module('resblock %d' % (i+3), BLOCK_Inception(convs[i], convs[i+1], 0.2))

    def forward(self,x):
        new_features =super(Residual_3D_Net_Inception, self).forward(x)
        return new_features


class BrainNet_3D_Inception(nn.Module):
    def __init__(self, in_channels= 4, convs=[30, 40, 40, 50, 150, 5]):
        super(BrainNet_3D_Inception, self).__init__()
        self.high_res_net = Residual_3D_Net_Inception(in_channels, convs)
        self.low_res_net  = Residual_3D_Net_Inception(in_channels, convs)

        self.features     = nn.Sequential(OrderedDict([]))
        self.features.add_module('final_block', BLOCK_Inception(2*convs[3], convs[4], 0.2, pad = 1))
        self.features.add_module('final_layer', _Layer(convs[4], convs[5], 0.2, pad = 1))

    def forward(self, high_res_vol, low_res_vol, pred_size=9):
        high_res_features = self.high_res_net(high_res_vol)
        low_res_features  = self.low_res_net(low_res_vol)
        # print ('high_res_features',high_res_features.size())
        # print ('low_res_features',low_res_features.size())
        size = high_res_features.size()

        # Upsample low_res_features
        # print (low_res_features.size(), high_res_features.size())
        # print ('low_res_upsample_features',low_res_upsample_features.size())
        low_res_upsample_features = F.upsample(low_res_features,(pred_size, pred_size, pred_size))
        concat = torch.cat([high_res_features, low_res_upsample_features], 1)
        concat = self.features(concat)
        return concat



# class Projection_Layer_Inception(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features, drop_rate=0, pad = 0):
#         super(Projection_Layer_Inception, self).__init__()
#         self.norm = nn.BatchNorm3d(num_input_features)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv = nn.Conv3d(num_input_features,num_output_features , kernel_size=1, stride=1, bias=False)

#         self.kernel_3x3 = nn.Conv3d(num_output_features,num_output_features , kernel_size=3, stride=1, bias=False, padding = 1)
#         self.kernel_5x5 = nn.Conv3d(num_output_features,num_output_features , kernel_size=5, stride=1, bias=False, padding = 2)
#         self.kernel_7x7 = nn.Conv3d(num_output_features,num_output_features , kernel_size=7, stride=1, bias=False, padding = 3)
#         self.drop_rate = drop_rate
#         self.pad = pad

#     def forward(self, x):
#         new_features = self.conv(self.relu(self.norm(x)))
#         new_features_3 = self.kernel_3x3(new_features)
#         new_features_5 = self.kernel_5x5(new_features)
#         new_features_7 = self.kernel_7x7(new_features)
#         new_featuress   = new_features_3 + new_features_5 + new_features_7
#         if self.drop_rate > 0:
#             new_featuress = F.dropout(new_featuress, p=self.drop_rate, training=self.training)
#         p3d = (-2,-2,-2, -2, -2, -2)

#         if not self.pad: new_featuress = F.pad(new_featuress, p3d)
#         else : new_featuress = new_featuress

#         return new_featuress

# class BLOCK_Inception(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features, drop_rate, pad = 0):
#         super(BLOCK_Inception, self).__init__()
#         self.layer1 = _Layer(num_input_features,num_output_features,drop_rate, pad)
#         self.layer2 = _Layer(num_output_features,num_output_features,drop_rate, pad)
#         self.pro = Projection_Layer_Inception(num_input_features, num_output_features, pad = pad)

#     def forward(self, x):
#         new_features = self.layer2(self.layer1(x))
#         num_input_features = x.size()[1]
#         projection_op = self.pro(x)
#         return torch.add(new_features, projection_op)

# class Residual_3D_Net_Inception(nn.Sequential):
#     def __init__(self,in_channels= 4, convs=[30, 40, 40, 50]):
#         super(Residual_3D_Net_Inception, self).__init__()
#         self.add_module('resblock 1', _Layer(in_channels, convs[0], 0.2))
#         self.add_module('resblock 2', _Layer(convs[0], convs[0], 0.2))
#         for i in range(3):
#             self.add_module('resblock %d' % (i+3), BLOCK_Inception(convs[i], convs[i+1], 0.2))

#     def forward(self,x):
#         new_features =super(Residual_3D_Net_Inception, self).forward(x)
#         return new_features


# class BrainNet_3D_Inception(nn.Module):
#     def __init__(self, in_channels= 4, convs=[30, 40, 40, 50, 150, 5]):
#         super(BrainNet_3D_Inception, self).__init__()
#         self.high_res_net = Residual_3D_Net_Inception(in_channels, convs)
#         self.low_res_net  = Residual_3D_Net_Inception(in_channels, convs)

#         self.features     = nn.Sequential(OrderedDict([]))
#         self.features.add_module('final_block', BLOCK_Inception(2*convs[3], convs[4], 0.2, pad = 1))
#         self.features.add_module('final_layer', _Layer(convs[4], convs[5], 0.2, pad = 1))

#     def forward(self, high_res_vol, low_res_vol):
#         high_res_features = self.high_res_net(high_res_vol)
#         low_res_features  = self.low_res_net(low_res_vol)
#         # print ('high_res_features',high_res_features.size())
#         # print ('low_res_features',low_res_features.size())
#         size = high_res_features.size()

#         #Upsample low_res_features
#         low_res_upsample_features = F.upsample(low_res_features,(9,9,9))
#         # print ('low_res_upsample_features',low_res_upsample_features.size())
#         concat = torch.cat([high_res_features, low_res_upsample_features], 1)
#         concat = self.features(concat)
#         return concat

if __name__ == '__main__':
    net = BrainNet_3D_Inception()
    print (net)
    high = Variable(torch.rand(3,4,25,25,25))
    low = Variable(torch.rand(3,4,19,19,19))
    print (net(high, low).size())
