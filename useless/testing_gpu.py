import tensorflow as tf
import torch

print("PyTorch GPU Availability:")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))

print("TensorFlow GPU Availability:")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(tf.config.list_physical_devices('GPU')) > 0:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
print("Built with CUDA: ", tf.test.is_built_with_cuda())
print("Is GPU available? ", tf.test.is_gpu_available())
