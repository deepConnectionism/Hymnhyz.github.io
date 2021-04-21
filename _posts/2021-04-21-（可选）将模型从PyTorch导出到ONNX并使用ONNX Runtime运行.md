在本教程中，我们描述了如何将PyTorch中定义的模型转换为ONNX格式，然后在ONNX Runtime中运行它。

ONNX Runtime是针对ONNX模型的以性能为中心的引擎，可跨多个平台和硬件（Windows，Linux和Mac，以及在CPU和GPU上）高效地进行推理。事实证明，ONNX Runtime可以显着提高多种模型的性能，[如此处所述](https://cloudblogs.microsoft.com/opensource/2019/05/22/onnx-runtime-machine-learning-inferencing-0-4-release)

对于本教程，您将需要安装ONNX和ONNX Runtime。您可以通过

    pip install onnx onnxruntime

获得ONNX和ONNX Runtime的二进制版本。请注意，ONNX运行时与Python 3.5至3.7版本兼容。

注意：本教程需要PyTorch master分支，可以按照此处的说明进行安装

    # Some standard imports
    import io
    import numpy as np

    from torch import nn
    import torch.utils.model_zoo as model_zoo
    import torch.onnx

超分辨率是提高图像，视频分辨率的一种方式，广泛用于图像处理或视频编辑中。在本教程中，我们将使用一个小的超分辨率模型。

首先，让我们在PyTorch中创建一个SuperResolution模型。该模型使用高效的子像素卷积层（描述在： “Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network” - Shi et al”）通过放大倍数来提高图像的分辨率。该模型期望图像的YCbCr的Y分量作为输入，并以超分辨率输出放大的Y分量。

该模型直接来自PyTorch的示例，未经修改：

    # Super Resolution model definition in PyTorch
    import torch.nn as nn
    import torch.nn.init as init


    class SuperResolutionNet(nn.Module):
        def __init__(self, upscale_factor, inplace=False):
            super(SuperResolutionNet, self).__init__()

            self.relu = nn.ReLU(inplace=inplace)
            self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
            self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
            self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
            self.conv4 = nn.Conv2d(32, upscale_factor ** 2, (3, 3), (1, 1), (1, 1))
            self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

            self._initialize_weights()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.pixel_shuffle(self.conv4(x))
            return x

        def _initialize_weights(self):
            init.orthogonal_(self.conv1.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv2.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv3.weight, init.calculate_gain('relu'))
            init.orthogonal_(self.conv4.weight)

    # Create the super-resolution model by using the above model definition.
    torch_model = SuperResolutionNet(upscale_factor=3)

通常，您现在将训练此模型。但是，在本教程中，我们将下载一些预训练的权重。请注意，此模型未经过充分训练以提供良好的准确性，在此仅用于演示目的。

在导出模型之前，必须先调用torch_model.eval（）或torch_model.train（False），以将模型转换为推理模式，这一点很重要。这是必需的，因为像dropout或batchnorm这样的运算符在推断和训练模式下的行为会有所不同。

    # Load pretrained model weights
    model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
    batch_size = 1    # just a random number

    # Initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))

    # set the model to inference mode
    torch_model.eval()

在PyTorch工作中导出模型是通过tracing或scripting。本教程将以通过tracing导出的模型为例。要导出模型，我们调用torch.onnx.export（）函数。这将执行模型，并记录使用什么运算符计算输出的轨迹。由于export运行模型，因此我们需要提供输入张量x。只要它是正确的类型和大小，其中的值就可以是随机的。请注意，除非指定为动态轴，否则输入尺寸将在导出的ONNX图形中固定为所有输入尺寸。在此示例中，我们使用输入batch_size 1导出模型，但随后在torch.onnx.export（）的dynamic_axes参数中将第一维指定为dynamic。因此，导出的模型将接受大小为[batch_size，1，224，224]的输入，其中batch_size可以是可变的。

要了解有关PyTorch导出界面的更多详细信息，请查看[torch.onnx](https://pytorch.org/docs/master/onnx.html)文档。

    # Input to the model
    x = torch.randn(batch_size, 1, 224, 224, requires_grad=True)
    torch_out = torch_model(x)

    # Export the model
    torch.onnx.export(torch_model,               # 模型正在运行 
                    x,                         # 模型输入（或用于多个输入的元组）
                    "super_resolution.onnx",   # 保存模型的位置（可以是文件或类似文件的对象）
                    export_params=True,        # 将训练后的参数权重存储在模型文件中
                    opset_version=10,          # ONNX版本以将模型导出到
                    do_constant_folding=True,  # 是否执行常量折叠以进行优化
                    input_names = ['input'],   # the model's input names
                    output_names = ['output'], # the model's output names
                    dynamic_axes={'input' : {0 : 'batch_size'},    # 可变长度轴 
                                'output' : {0 : 'batch_size'}})

我们还计算了模型后的输出torch_out，我们将使用它来验证导出的模型在ONNX Runtime中运行时是否计算出相同的值。

但是，在通过ONNX Runtime验证模型的输出之前，我们将使用ONNX的API检查ONNX模型。首先，

    onnx.load（"super_resolution.onnx"）
将加载保存的模型并输出onnx.ModelProto结构（用于捆绑ML模型的顶级文件/容器格式。有关onnx.proto文档的[更多信息](https://github.com/onnx/onnx/blob/master/onnx/onnx.proto)。 然后，onnx.checker.check_model(onnx_model)将验证模型的结构并确认模型具有有效的架构。通过检查模型的版本，图形的结构以及节点及其输入和输出，可以验证ONNX图的有效性。

    import onnx

    onnx_model = onnx.load("super_resolution.onnx")
    onnx.checker.check_model(onnx_model)

现在，我们使用ONNX Runtime的Python API计算输出。这部分通常可以在单独的过程中或在另一台机器上完成，但是我们将继续同一过程，以便我们可以验证ONNX Runtime和PyTorch正在为网络计算相同的值。

为了使用ONNX Runtime运行模型，我们需要使用所选的配置参数为模型创建一个推理会话（此处使用默认配置）。 创建会话后，我们将使用run()API评估模型。 此调用的输出是一个列表，其中包含由ONNX Runtime计算的模型的输出。

    import onnxruntime

    ort_session = onnxruntime.InferenceSession("super_resolution.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

我们应该看到PyTorch和ONNX Runtime的输出在数值上与给定的精度匹配（rtol = 1e-03和atol = 1e-05）。附带说明一下，如果它们不匹配，则说明ONNX导出器中存在问题，因此在这种情况下，请与我们联系。

## 使用ONNX Runtime在图像上运行模型

到目前为止，我们已经从PyTorch导出了一个模型，并展示了如何在虚拟张量作为输入的情况下在ONNX Runtime中加载和运行该模型。

在本教程中，我们将使用广泛使用的著名猫图像，如下图所示

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="assets/images/cat_224x224.jpg">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;"> </div>
</center>

首先，让我们加载图片，然后使用标准的PIL python库对其进行预处理。请注意，这种预处理是处理数据以训练/测试神经网络的标准做法。

我们首先调整图像大小以适合模型输入的大小（224x224）。然后，我们将图像分为其Y，Cb和Cr分量。这些分量代表灰度图像（Y），以及蓝差blue-difference（Cb）和红差red-difference（Cr）色度分量。Y分量对人眼更为敏感，我们对将要转换的Y分量感兴趣。提取Y分量后，我们将其转换为张量，这将是模型的输入。

    from PIL import Image
    import torchvision.transforms as transforms

    img = Image.open("./_static/img/cat.jpg")

    resize = transforms.Resize([224, 224])
    img = resize(img)

    img_ycbcr = img.convert('YCbCr')
    img_y, img_cb, img_cr = img_ycbcr.split()

    to_tensor = transforms.ToTensor()
    img_y = to_tensor(img_y)
    img_y.unsqueeze_(0)

现在，作为下一步，让我们使用表示已调整大小的灰度猫图像的张量，并按照先前的说明在ONNX Runtime中运行超分辨率模型。

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_y)}
    ort_outs = ort_session.run(None, ort_inputs)
    img_out_y = ort_outs[0]

此时，模型的输出为张量。现在，我们将处理模型的输出，以根据输出张量构造最终的输出图像，并保存图像。后处理步骤已从[此处](https://github.com/pytorch/examples/blob/master/super_resolution/super_resolve.py)的超分辨率模型的PyTorch实现中采用。

    img_out_y = Image.fromarray(np.uint8((img_out_y[0] * 255.0).clip(0, 255)[0]), mode='L')

    # get the output image follow post-processing step from PyTorch implementation
    final_img = Image.merge(
        "YCbCr", [
            img_out_y,
            img_cb.resize(img_out_y.size, Image.BICUBIC),
            img_cr.resize(img_out_y.size, Image.BICUBIC),
        ]).convert("RGB")

    # Save the image, we will compare this with the output image from mobile device
    final_img.save("./_static/img/cat_superres_with_ort.jpg")

ONNX Runtime是跨平台引擎，您可以跨多个平台在CPU和GPU上运行它。

还可以使用Azure机器学习服务将ONNX Runtime部署到云中以进行模型推断。[更多信息在这里](https://docs.microsoft.com/en-us/azure/machine-learning/service/concept-onnx)。

[点击此处](https://github.com/microsoft/onnxruntime#high-performance)，详细了解ONNX运行时的性能。

有关ONNX Runtime的更多信息，请点击[此处](https://github.com/microsoft/onnxruntime)。

